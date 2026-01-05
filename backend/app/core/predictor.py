"""
Unified prediction engine for WHAT/WHEN/HOW/WHY predictions.
"""
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Any

from app.core.xes_parser import Event
from app.core.prefix_generator import PrefixSample
from app.core.trainer import ProcessPredictor
from app.core.time_confidence import calculate_aggregate_confidence


@dataclass
class PredictionResult:
    """Complete prediction result with all dimensions."""
    # WHAT - Next activity
    next_activity: str
    next_activity_probability: float
    next_activity_calibrated: float
    next_activity_confidence: str
    alternative_activities: List[Tuple[str, float]]
    
    # WHEN - Timing
    expected_remaining_time_hours: float
    time_lower_bound_hours: float
    time_upper_bound_hours: float
    time_confidence: str
    
    # HOW - Outcome
    predicted_outcome: str
    outcome_probability: float
    outcome_calibrated: float
    outcome_confidence: str
    outcome_distribution: Dict[str, float]
    
    # WHY - Explanation
    feature_importance: List[Tuple[str, float]]
    risk_factors: List[str]
    
    # Aggregate confidence
    aggregate_score: float
    aggregate_level: str
    confidence_flags: List[str]
    
    # Recommendations (optional)
    recommendations: Optional[List[str]] = None


class PredictionEngine:
    """
    Unified prediction engine for process mining.
    
    Provides comprehensive predictions including:
    - WHAT: Next activity prediction
    - WHEN: Remaining time with confidence intervals
    - HOW: Outcome prediction
    - WHY: Feature importance and risk factors
    """
    
    def __init__(self, predictor: ProcessPredictor):
        self.predictor = predictor
    
    def predict(self, events: List[Event]) -> PredictionResult:
        """
        Generate comprehensive prediction for a running case.
        
        Args:
            events: List of events in the running case
            
        Returns:
            PredictionResult with all prediction dimensions
        """
        if not events:
            raise ValueError("Cannot predict on empty event list")
        
        # Create prefix sample
        prefix = PrefixSample(
            case_id="runtime",
            prefix_events=events,
            target_activity="",
            target_outcome=None,
            remaining_time=0
        )
        
        # Extract features
        features = self.predictor.feature_engineer.transform(prefix)
        X = features.reshape(1, -1)
        
        # ─────────────────────────────────────────────────────────────────────
        # WHAT - Next activity prediction
        # ─────────────────────────────────────────────────────────────────────
        activity_preds, activity_probs = self.predictor.predict_next_activity(X)
        calibrated_probs = activity_probs[0]
        
        sorted_indices = np.argsort(calibrated_probs)[::-1]
        
        next_activity = self.predictor.label_decoder.get(sorted_indices[0], "Unknown")
        next_prob = float(calibrated_probs[sorted_indices[0]])
        
        # Get raw probability for comparison
        raw_probs = self.predictor.next_activity_model.predict_proba(X)[0]
        next_prob_raw = float(raw_probs[sorted_indices[0]])
        
        # Alternatives (top 3 excluding the best)
        alternatives = [
            (self.predictor.label_decoder.get(i, f"Unknown_{i}"), float(calibrated_probs[i]))
            for i in sorted_indices[1:4]
        ]
        
        # Confidence level
        activity_confidence = self._get_confidence_level(next_prob)
        
        # ─────────────────────────────────────────────────────────────────────
        # WHEN - Time prediction
        # ─────────────────────────────────────────────────────────────────────
        time_predictions = self.predictor.predict_time(X)
        time_pred = time_predictions[0]
        
        # ─────────────────────────────────────────────────────────────────────
        # HOW - Outcome prediction
        # ─────────────────────────────────────────────────────────────────────
        outcome_preds, outcome_probs = self.predictor.predict_outcome(X)
        outcome_calibrated = outcome_probs[0]
        
        outcome_idx = int(np.argmax(outcome_calibrated))
        predicted_outcome = self.predictor.outcome_decoder.get(outcome_idx, "Unknown")
        outcome_prob = float(outcome_calibrated[outcome_idx])
        
        # Raw outcome probability
        if self.predictor.outcome_model is not None:
            raw_outcome_probs = self.predictor.outcome_model.predict_proba(X)[0]
            outcome_prob_raw = float(raw_outcome_probs[outcome_idx])
        else:
            outcome_prob_raw = outcome_prob
        
        outcome_confidence = self._get_confidence_level(outcome_prob)
        
        # Outcome distribution
        outcome_distribution = {
            self.predictor.outcome_decoder.get(i, f"Outcome_{i}"): float(p)
            for i, p in enumerate(outcome_calibrated)
        }
        
        # ─────────────────────────────────────────────────────────────────────
        # WHY - Explanation
        # ─────────────────────────────────────────────────────────────────────
        feature_importance = self.predictor.get_feature_importance()[:5]
        risk_factors = self._identify_risks(features, events)
        
        # ─────────────────────────────────────────────────────────────────────
        # Aggregate Confidence
        # ─────────────────────────────────────────────────────────────────────
        median_time = self.predictor.time_quantile_model.median_time / 3600
        aggregate = calculate_aggregate_confidence(
            next_activity_prob=next_prob,
            outcome_prob=outcome_prob,
            time_interval_width=time_pred.interval_width_hours,
            median_process_time=median_time
        )
        
        return PredictionResult(
            next_activity=next_activity,
            next_activity_probability=next_prob_raw,
            next_activity_calibrated=next_prob,
            next_activity_confidence=activity_confidence,
            alternative_activities=alternatives,
            expected_remaining_time_hours=time_pred.point_estimate,
            time_lower_bound_hours=time_pred.lower_bound,
            time_upper_bound_hours=time_pred.upper_bound,
            time_confidence=time_pred.confidence_level,
            predicted_outcome=predicted_outcome,
            outcome_probability=outcome_prob_raw,
            outcome_calibrated=outcome_prob,
            outcome_confidence=outcome_confidence,
            outcome_distribution=outcome_distribution,
            feature_importance=feature_importance,
            risk_factors=risk_factors,
            aggregate_score=aggregate.overall_score,
            aggregate_level=aggregate.overall_level,
            confidence_flags=aggregate.flags
        )
    
    def predict_batch(self, events_list: List[List[Event]]) -> List[PredictionResult]:
        """Predict for multiple cases."""
        return [self.predict(events) for events in events_list]
    
    def _get_confidence_level(self, prob: float) -> str:
        """Classify probability into confidence level."""
        if prob >= 0.75:
            return 'HIGH'
        elif prob >= 0.50:
            return 'MEDIUM'
        else:
            return 'LOW'
    
    def _identify_risks(self, features: np.ndarray, events: List[Event]) -> List[str]:
        """
        Simple rule-based risk identification.
        
        Based on feature values that indicate potential issues.
        """
        risks = []
        
        # Check elapsed time (feature 0, normalized)
        if features[0] > 2.0:
            risks.append("Case duration exceeds typical by 2x")
        
        # Check loop count (feature 18)
        if len(features) > 18 and features[18] > 3:
            risks.append(f"Multiple activity repetitions detected ({int(features[18])} loops)")
        
        # Check transition probability (feature 16)
        if len(features) > 16 and features[16] < 0.1:
            risks.append("Unusual activity transition (rarely seen in training)")
        
        # Check time since last event (feature 1, in hours)
        if features[1] > 24:
            risks.append(f"Long idle time: {features[1]:.1f} hours since last activity")
        
        # Check activity rarity (feature 21)
        if len(features) > 21 and features[21] > 50:
            risks.append("Current activity is rare in this process")
        
        return risks
    
    def get_explanation(self, events: List[Event]) -> Dict[str, Any]:
        """
        Get detailed explanation for a prediction.
        
        Returns feature values and their contributions.
        """
        prefix = PrefixSample(
            case_id="runtime",
            prefix_events=events,
            target_activity="",
            target_outcome=None,
            remaining_time=0
        )
        
        features = self.predictor.feature_engineer.transform(prefix)
        feature_names = self.predictor.feature_engineer.feature_names
        
        return {
            "features": [
                {"name": name, "value": float(val)}
                for name, val in zip(feature_names, features)
            ],
            "importance": [
                {"name": name, "importance": float(imp)}
                for name, imp in self.predictor.get_feature_importance()[:10]
            ]
        }
