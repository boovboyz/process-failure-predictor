"""
Model evaluation on test set.
"""
import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    precision_score, recall_score, confusion_matrix,
    mean_absolute_error, mean_squared_error
)
from typing import List, Dict, Any
from dataclasses import dataclass, asdict

from app.core.xes_parser import Trace
from app.core.prefix_generator import generate_prefixes, generate_prefixes_at_percentage, PrefixSample
from app.core.trainer import ProcessPredictor
from app.core.calibration import ConfidenceMetrics


@dataclass
class EvaluationResults:
    """Complete evaluation results."""
    # Next Activity metrics
    next_activity_accuracy: float
    next_activity_top3_accuracy: float
    next_activity_macro_f1: float
    next_activity_confusion_matrix: List[List[int]]
    next_activity_class_labels: List[str]
    
    # Outcome metrics
    outcome_auc_roc: float
    outcome_precision: float
    outcome_recall: float
    outcome_f1: float
    
    # Time prediction metrics
    time_mae_hours: float
    time_rmse_hours: float
    time_mape: float
    time_coverage: float
    
    # Early detection (accuracy at different prefix lengths)
    early_detection: Dict[str, float]
    
    # Calibration metrics
    expected_calibration_error: float
    
    # Sample predictions for UI display
    sample_predictions: List[Dict[str, Any]]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ModelEvaluator:
    """Evaluates trained models on test set."""
    
    def __init__(self, predictor: ProcessPredictor):
        self.predictor = predictor
    
    def evaluate(self, test_traces: List[Trace]) -> EvaluationResults:
        """
        Run full evaluation on test set.
        
        Args:
            test_traces: List of test traces
            
        Returns:
            EvaluationResults with all metrics
        """
        # Generate test prefixes
        test_prefixes = generate_prefixes(test_traces)
        
        if not test_prefixes:
            raise ValueError("No valid test prefixes generated")
        
        # Extract features
        X_test = self.predictor.feature_engineer.transform_batch(test_prefixes)
        
        # True labels
        y_activity_true = np.array([
            self.predictor.label_encoder.get(p.target_activity, 0)
            for p in test_prefixes
        ])
        y_outcome_true = np.array([
            self.predictor.outcome_encoder.get(p.target_outcome, 0)
            for p in test_prefixes
        ])
        y_time_true = np.array([p.remaining_time for p in test_prefixes])
        
        # ─────────────────────────────────────────────────────────────────────
        # Next Activity Evaluation
        # ─────────────────────────────────────────────────────────────────────
        activity_probs = self.predictor.next_activity_model.predict_proba(X_test)
        activity_preds = np.argmax(activity_probs, axis=1)
        
        # Top-3 accuracy
        top3_correct = sum(
            true in np.argsort(probs)[-3:]
            for true, probs in zip(y_activity_true, activity_probs)
        )
        top3_accuracy = top3_correct / len(y_activity_true)
        
        # Confusion matrix
        unique_labels = sorted(set(y_activity_true))
        cm = confusion_matrix(y_activity_true, activity_preds, labels=unique_labels)
        
        # ECE
        ece = ConfidenceMetrics.expected_calibration_error(y_activity_true, activity_probs)
        
        # ─────────────────────────────────────────────────────────────────────
        # Outcome Evaluation
        # ─────────────────────────────────────────────────────────────────────
        if self.predictor.outcome_model is not None:
            outcome_probs = self.predictor.outcome_model.predict_proba(X_test)
            outcome_preds = np.argmax(outcome_probs, axis=1)
            
            # AUC-ROC
            try:
                if outcome_probs.shape[1] == 2:
                    auc = roc_auc_score(y_outcome_true, outcome_probs[:, 1])
                else:
                    auc = roc_auc_score(y_outcome_true, outcome_probs, multi_class='ovr')
            except Exception:
                auc = 0.5
            
            outcome_precision = precision_score(y_outcome_true, outcome_preds, average='weighted', zero_division=0)
            outcome_recall = recall_score(y_outcome_true, outcome_preds, average='weighted', zero_division=0)
            outcome_f1 = f1_score(y_outcome_true, outcome_preds, average='weighted', zero_division=0)
        else:
            auc = 0.5
            outcome_precision = 0.0
            outcome_recall = 0.0
            outcome_f1 = 0.0
        
        # ─────────────────────────────────────────────────────────────────────
        # Time Prediction Evaluation
        # ─────────────────────────────────────────────────────────────────────
        time_preds_seconds = self.predictor.time_model.predict(X_test)
        time_preds_hours = time_preds_seconds / 3600
        y_time_hours = y_time_true / 3600
        
        time_mae = mean_absolute_error(y_time_hours, time_preds_hours)
        time_rmse = np.sqrt(mean_squared_error(y_time_hours, time_preds_hours))
        
        # MAPE (avoiding division by zero)
        mape = np.mean(np.abs((y_time_hours - time_preds_hours) / (y_time_hours + 1e-6))) * 100
        
        # Coverage
        time_coverage = self.predictor.time_quantile_model.coverage_score(X_test, y_time_true)
        
        # ─────────────────────────────────────────────────────────────────────
        # Early Detection Analysis
        # ─────────────────────────────────────────────────────────────────────
        early_detection = self._evaluate_early_detection(test_traces)
        
        # ─────────────────────────────────────────────────────────────────────
        # Sample Predictions for UI
        # ─────────────────────────────────────────────────────────────────────
        samples = self._get_sample_predictions(test_prefixes[:10], X_test[:10])
        
        return EvaluationResults(
            next_activity_accuracy=accuracy_score(y_activity_true, activity_preds),
            next_activity_top3_accuracy=top3_accuracy,
            next_activity_macro_f1=f1_score(y_activity_true, activity_preds, average='macro', zero_division=0),
            next_activity_confusion_matrix=cm.tolist(),
            next_activity_class_labels=[self.predictor.label_decoder.get(l, str(l)) for l in unique_labels],
            outcome_auc_roc=auc,
            outcome_precision=outcome_precision,
            outcome_recall=outcome_recall,
            outcome_f1=outcome_f1,
            time_mae_hours=time_mae,
            time_rmse_hours=time_rmse,
            time_mape=mape,
            time_coverage=time_coverage,
            early_detection=early_detection,
            expected_calibration_error=ece,
            sample_predictions=samples
        )
    
    def _evaluate_early_detection(self, test_traces: List[Trace]) -> Dict[str, float]:
        """Evaluate prediction accuracy at different prefix completion percentages."""
        prefix_by_pct = generate_prefixes_at_percentage(test_traces, [25, 50, 75])
        results = {}
        
        for pct, prefixes in prefix_by_pct.items():
            if not prefixes:
                results[f"{pct}%"] = 0.0
                continue
            
            X = self.predictor.feature_engineer.transform_batch(prefixes)
            
            # Predict
            probs = self.predictor.next_activity_model.predict_proba(X)
            preds = np.argmax(probs, axis=1)
            
            # True labels
            y_true = np.array([
                self.predictor.label_encoder.get(p.target_activity, 0)
                for p in prefixes
            ])
            
            accuracy = (preds == y_true).mean()
            results[f"{pct}%"] = float(accuracy)
        
        return results
    
    def _get_sample_predictions(self, prefixes: List[PrefixSample],
                                X: np.ndarray) -> List[Dict[str, Any]]:
        """Get sample predictions for UI display."""
        samples = []
        
        for i, prefix in enumerate(prefixes):
            probs = self.predictor.next_activity_model.predict_proba(X[i:i+1])[0]
            pred_idx = np.argmax(probs)
            pred_activity = self.predictor.label_decoder.get(pred_idx, str(pred_idx))
            
            samples.append({
                "case_id": prefix.case_id,
                "prefix_length": len(prefix.prefix_events),
                "last_activity": prefix.prefix_events[-1].activity,
                "predicted_next": pred_activity,
                "predicted_probability": float(probs[pred_idx]),
                "actual_next": prefix.target_activity,
                "correct": pred_activity == prefix.target_activity
            })
        
        return samples
    
    def get_confusion_matrix_details(self, test_traces: List[Trace]) -> Dict[str, Any]:
        """Get detailed confusion matrix with labels."""
        test_prefixes = generate_prefixes(test_traces)
        X_test = self.predictor.feature_engineer.transform_batch(test_prefixes)
        
        y_true = np.array([
            self.predictor.label_encoder.get(p.target_activity, 0)
            for p in test_prefixes
        ])
        
        probs = self.predictor.next_activity_model.predict_proba(X_test)
        y_pred = np.argmax(probs, axis=1)
        
        unique_labels = sorted(set(y_true))
        cm = confusion_matrix(y_true, y_pred, labels=unique_labels)
        
        return {
            "matrix": cm.tolist(),
            "labels": [self.predictor.label_decoder.get(l, str(l)) for l in unique_labels],
            "total_predictions": len(y_true)
        }
