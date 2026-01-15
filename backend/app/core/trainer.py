"""
Model training pipeline for process prediction.
Trains XGBoost or LightGBM models for next-activity, outcome, and time prediction.
"""
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from typing import List, Dict, Optional, Tuple, Literal
import joblib
import time
from pathlib import Path

# Optional LightGBM support
try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

from app.core.xes_parser import Trace
from app.core.prefix_generator import generate_prefixes, PrefixSample
from app.core.feature_engineer import FeatureEngineer
from app.core.calibration import ProbabilityCalibrator
from app.core.time_confidence import QuantileTimePredictor


# Model type alias
ModelType = Literal["xgboost", "lightgbm"]


class ProcessPredictor:
    """
    Complete prediction pipeline for process mining.
    
    Trains and manages:
    - Next activity prediction (multi-class classification)
    - Outcome prediction (multi-class classification)
    - Remaining time prediction (quantile regression)
    
    Supports both XGBoost and LightGBM backends.
    """
    
    def __init__(self, model_type: ModelType = "xgboost"):
        """
        Initialize the predictor.
        
        Args:
            model_type: Model backend to use ("xgboost" or "lightgbm")
        """
        self.model_type = model_type
        
        if model_type == "lightgbm" and not HAS_LIGHTGBM:
            raise ImportError("LightGBM not installed. Run: pip install lightgbm")
        
        self.feature_engineer = FeatureEngineer()
        self.next_activity_model = None
        self.outcome_model = None
        self.time_model = None
        self.time_quantile_model = QuantileTimePredictor()
        
        # Calibrators
        self.activity_calibrator = ProbabilityCalibrator()
        self.outcome_calibrator = ProbabilityCalibrator()
        
        # Label encoders
        self.label_encoder: Dict[str, int] = {}
        self.label_decoder: Dict[int, str] = {}
        self.outcome_encoder: Dict[str, int] = {}
        self.outcome_decoder: Dict[int, str] = {}
        
        self.is_trained = False
    
    def _create_classifier(self, n_classes: int, is_binary: bool = False):
        """Create a classifier based on model_type."""
        if self.model_type == "lightgbm":
            return lgb.LGBMClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                objective='binary' if is_binary else 'multiclass',
                num_class=None if is_binary else n_classes,
                metric='binary_logloss' if is_binary else 'multi_logloss',
                random_state=42,
                verbose=-1,
                force_col_wise=True,
            )
        else:  # xgboost
            return xgb.XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                objective='binary:logistic' if is_binary else 'multi:softprob',
                num_class=None if is_binary else n_classes,
                eval_metric='logloss' if is_binary else 'mlogloss',
                early_stopping_rounds=20,
                random_state=42,
                verbosity=0,
            )
    
    def _create_regressor(self):
        """Create a regressor based on model_type."""
        if self.model_type == "lightgbm":
            return lgb.LGBMRegressor(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                objective='regression',
                metric='mae',
                random_state=42,
                verbose=-1,
                force_col_wise=True,
            )
        else:  # xgboost
            return xgb.XGBRegressor(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                objective='reg:squarederror',
                eval_metric='mae',
                early_stopping_rounds=20,
                random_state=42,
                verbosity=0,
            )
    
    def _create_tuned_classifier(self, params: Dict, n_classes: int, is_binary: bool = False):
        """Create a classifier with tuned hyperparameters."""
        params = {**params, 'random_state': 42}
        
        if self.model_type == "lightgbm":
            params['verbose'] = -1
            params['force_col_wise'] = True
            if is_binary:
                params['objective'] = 'binary'
                params['metric'] = 'binary_logloss'
            else:
                params['objective'] = 'multiclass'
                params['num_class'] = n_classes
                params['metric'] = 'multi_logloss'
            return lgb.LGBMClassifier(**params)
        else:
            params['verbosity'] = 0
            if is_binary:
                params['objective'] = 'binary:logistic'
                params['eval_metric'] = 'logloss'
            else:
                params['objective'] = 'multi:softprob'
                params['num_class'] = n_classes
                params['eval_metric'] = 'mlogloss'
            return xgb.XGBClassifier(**params)
    
    def _create_tuned_regressor(self, params: Dict):
        """Create a regressor with tuned hyperparameters."""
        params = {**params, 'random_state': 42}
        
        if self.model_type == "lightgbm":
            params['verbose'] = -1
            params['force_col_wise'] = True
            params['objective'] = 'regression'
            params['metric'] = 'mae'
            return lgb.LGBMRegressor(**params)
        else:
            params['verbosity'] = 0
            params['objective'] = 'reg:squarederror'
            params['eval_metric'] = 'mae'
            return xgb.XGBRegressor(**params)
    
    def train(
        self,
        train_traces: List[Trace], 
        val_ratio: float = 0.2,
        auto_tune: bool = False,
        tune_trials: int = 30,
    ) -> Dict[str, Dict[str, float]]:
        """
        Train all prediction models.
        
        Args:
            train_traces: List of training traces
            val_ratio: Ratio of training data to use for validation
            auto_tune: Whether to run hyperparameter tuning before training
            tune_trials: Number of Optuna trials for hyperparameter tuning
            
        Returns:
            Dictionary of metrics per task
        """
        start_time = time.time()
        
        # Fit feature engineer
        self.feature_engineer.fit(train_traces)
        
        # Generate prefixes
        prefixes = generate_prefixes(train_traces)
        
        if not prefixes:
            raise ValueError("No valid prefixes generated from training traces")
        
        # Build feature matrix
        X = self.feature_engineer.transform_batch(prefixes)
        
        # Prepare labels for next activity
        activities = list(set(p.target_activity for p in prefixes))
        self.label_encoder = {a: i for i, a in enumerate(sorted(activities))}
        self.label_decoder = {i: a for a, i in self.label_encoder.items()}
        
        y_activity = np.array([
            self.label_encoder.get(p.target_activity, 0) 
            for p in prefixes
        ])
        
        # Prepare labels for outcome
        outcomes = list(set(p.target_outcome for p in prefixes if p.target_outcome))
        self.outcome_encoder = {o: i for i, o in enumerate(sorted(outcomes))}
        self.outcome_decoder = {i: o for o, i in self.outcome_encoder.items()}
        
        y_outcome = np.array([
            self.outcome_encoder.get(p.target_outcome, 0) 
            for p in prefixes
        ])
        
        # Prepare time targets
        y_time = np.array([p.remaining_time for p in prefixes])
        
        # Train/validation split (within training data)
        X_train, X_val, y_act_train, y_act_val, y_out_train, y_out_val, y_time_train, y_time_val = \
            train_test_split(
                X, y_activity, y_outcome, y_time,
                test_size=val_ratio, random_state=42
            )
        
        metrics = {}
        
        # ─────────────────────────────────────────────────────────────────────
        # Optional Hyperparameter Tuning
        # ─────────────────────────────────────────────────────────────────────
        tuned_params = {}
        if auto_tune:
            from app.core.hyperparameter_tuning import HyperparameterTuner
            
            try:
                # Tune for next activity prediction
                activity_tuner = HyperparameterTuner(
                    model_type=self.model_type,
                    task_type="classification",
                    n_trials=tune_trials,
                    metric="accuracy",
                )
                tuned_params['activity'] = activity_tuner.tune(
                    X_train, y_act_train,
                    n_classes=len(activities),
                )
                
                # Tune for outcome prediction if applicable
                if len(outcomes) > 1:
                    outcome_tuner = HyperparameterTuner(
                        model_type=self.model_type,
                        task_type="classification",
                        n_trials=tune_trials,
                        metric="accuracy",
                    )
                    tuned_params['outcome'] = outcome_tuner.tune(
                        X_train, y_out_train,
                        n_classes=len(outcomes),
                    )
                
                # Tune for time prediction
                time_tuner = HyperparameterTuner(
                    model_type=self.model_type,
                    task_type="regression",
                    n_trials=tune_trials,
                    metric="mae",
                )
                tuned_params['time'] = time_tuner.tune(X_train, y_time_train)
                
                metrics['tuned_params'] = tuned_params
            except Exception as e:
                # Graceful fallback: if tuning fails, use default params
                print(f"⚠ Hyperparameter tuning failed: {e}. Using default parameters.")
                tuned_params = {}
                metrics['tuning_error'] = str(e)
        
        # ─────────────────────────────────────────────────────────────────────
        # Train Next Activity Model (multi-class)
        # ─────────────────────────────────────────────────────────────────────
        if auto_tune and 'activity' in tuned_params:
            self.next_activity_model = self._create_tuned_classifier(
                tuned_params['activity'],
                n_classes=len(activities),
                is_binary=False
            )
        else:
            self.next_activity_model = self._create_classifier(
                n_classes=len(activities),
                is_binary=False
            )
        
        if self.model_type == "lightgbm":
            self.next_activity_model.fit(X_train, y_act_train)
        else:
            self.next_activity_model.fit(
                X_train, y_act_train,
                eval_set=[(X_val, y_act_val)],
                verbose=False
            )
        
        # Evaluate and calibrate
        val_probs = self.next_activity_model.predict_proba(X_val)
        val_preds = np.argmax(val_probs, axis=1)
        activity_accuracy = (val_preds == y_act_val).mean()
        
        # Fit calibrator
        self.activity_calibrator.fit(y_act_val, val_probs)
        
        metrics['next_activity'] = {
            'accuracy': float(activity_accuracy),
            'n_classes': len(activities),
        }
        
        # ─────────────────────────────────────────────────────────────────────
        # Train Outcome Model (multi-class)
        # ─────────────────────────────────────────────────────────────────────
        if len(outcomes) > 1:
            if auto_tune and 'outcome' in tuned_params:
                self.outcome_model = self._create_tuned_classifier(
                    tuned_params['outcome'],
                    n_classes=len(outcomes),
                    is_binary=(len(outcomes) == 2)
                )
            else:
                self.outcome_model = self._create_classifier(
                    n_classes=len(outcomes),
                    is_binary=(len(outcomes) == 2)
                )
            
            if self.model_type == "lightgbm":
                self.outcome_model.fit(X_train, y_out_train)
            else:
                self.outcome_model.fit(
                    X_train, y_out_train,
                    eval_set=[(X_val, y_out_val)],
                    verbose=False
                )
            
            # Evaluate and calibrate
            out_probs = self.outcome_model.predict_proba(X_val)
            out_preds = np.argmax(out_probs, axis=1) if len(outcomes) > 2 else (out_probs[:, 1] > 0.5).astype(int)
            outcome_accuracy = (out_preds == y_out_val).mean()
            
            self.outcome_calibrator.fit(y_out_val, out_probs)
            
            metrics['outcome'] = {
                'accuracy': float(outcome_accuracy),
                'n_classes': len(outcomes),
            }
        else:
            metrics['outcome'] = {'accuracy': 1.0, 'n_classes': 1}
        
        # ─────────────────────────────────────────────────────────────────────
        # Train Time Model (regression + quantile)
        # ─────────────────────────────────────────────────────────────────────
        if auto_tune and 'time' in tuned_params:
            self.time_model = self._create_tuned_regressor(tuned_params['time'])
        else:
            self.time_model = self._create_regressor()
        
        if self.model_type == "lightgbm":
            self.time_model.fit(X_train, y_time_train)
        else:
            self.time_model.fit(
                X_train, y_time_train,
                eval_set=[(X_val, y_time_val)],
                verbose=False
            )
        
        # Train quantile models for confidence intervals
        self.time_quantile_model.fit(X_train, y_time_train)
        
        # Evaluate
        time_preds = self.time_model.predict(X_val)
        time_mae = np.mean(np.abs(time_preds - y_time_val))
        time_coverage = self.time_quantile_model.coverage_score(X_val, y_time_val)
        
        metrics['time'] = {
            'mae_seconds': float(time_mae),
            'mae_hours': float(time_mae / 3600),
            'coverage': float(time_coverage),
        }
        
        training_time = time.time() - start_time
        metrics['training_time_seconds'] = training_time
        
        self.is_trained = True
        
        return metrics
    
    def predict_next_activity(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict next activity.
        
        Returns:
            Tuple of (predicted_labels, probabilities)
        """
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        probs = self.next_activity_model.predict_proba(X)
        calibrated = self.activity_calibrator.calibrate(probs)
        preds = np.argmax(calibrated, axis=1)
        
        return preds, calibrated
    
    def predict_outcome(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict outcome.
        
        Returns:
            Tuple of (predicted_labels, probabilities)
        """
        if not self.is_trained or self.outcome_model is None:
            return np.zeros(len(X)), np.zeros((len(X), 1))
        
        probs = self.outcome_model.predict_proba(X)
        calibrated = self.outcome_calibrator.calibrate(probs)
        preds = np.argmax(calibrated, axis=1)
        
        return preds, calibrated
    
    def predict_time(self, X: np.ndarray):
        """
        Predict remaining time with confidence intervals.
        
        Returns:
            List of TimePredictionWithCI objects
        """
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        return self.time_quantile_model.predict(X)
    
    def get_feature_importance(self) -> List[Tuple[str, float]]:
        """Get feature importance from next activity model."""
        if not self.is_trained:
            return []
        
        feature_names = self.feature_engineer.feature_names
        importances = self.next_activity_model.feature_importances_
        
        return sorted(
            zip(feature_names, importances),
            key=lambda x: x[1],
            reverse=True
        )
    
    def save(self, path: str):
        """Save all models and state."""
        save_data = {
            'model_type': self.model_type,  # Persist model type
            'feature_engineer': self.feature_engineer.get_state(),
            'next_activity_model': self.next_activity_model,
            'outcome_model': self.outcome_model,
            'time_model': self.time_model,
            'time_quantile_model': {
                'median': self.time_quantile_model.model_median,
                'lower': self.time_quantile_model.model_lower,
                'upper': self.time_quantile_model.model_upper,
                'state': self.time_quantile_model.get_state(),
            },
            'activity_calibrator': self.activity_calibrator,
            'outcome_calibrator': self.outcome_calibrator,
            'label_encoder': self.label_encoder,
            'label_decoder': self.label_decoder,
            'outcome_encoder': self.outcome_encoder,
            'outcome_decoder': self.outcome_decoder,
            'is_trained': self.is_trained,
        }
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(save_data, path)
    
    @classmethod
    def load(cls, path: str) -> 'ProcessPredictor':
        """Load trained predictor."""
        data = joblib.load(path)
        
        # Restore model_type if saved, default to xgboost for backwards compatibility
        model_type = data.get('model_type', 'xgboost')
        predictor = cls(model_type=model_type)
        predictor.feature_engineer.set_state(data['feature_engineer'])
        predictor.next_activity_model = data['next_activity_model']
        predictor.outcome_model = data['outcome_model']
        predictor.time_model = data['time_model']
        
        # Restore quantile model
        predictor.time_quantile_model.model_median = data['time_quantile_model']['median']
        predictor.time_quantile_model.model_lower = data['time_quantile_model']['lower']
        predictor.time_quantile_model.model_upper = data['time_quantile_model']['upper']
        predictor.time_quantile_model.set_state(data['time_quantile_model']['state'])
        
        predictor.activity_calibrator = data['activity_calibrator']
        predictor.outcome_calibrator = data['outcome_calibrator']
        predictor.label_encoder = data['label_encoder']
        predictor.label_decoder = data['label_decoder']
        predictor.outcome_encoder = data['outcome_encoder']
        predictor.outcome_decoder = data['outcome_decoder']
        predictor.is_trained = data['is_trained']
        
        return predictor
