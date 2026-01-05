"""
Time prediction with confidence intervals using quantile regression.
"""
import numpy as np
import xgboost as xgb
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class TimePredictionWithCI:
    """Time prediction with confidence interval."""
    point_estimate: float      # Expected remaining time (hours)
    lower_bound: float         # 10th percentile
    upper_bound: float         # 90th percentile
    confidence_level: str      # HIGH, MEDIUM, LOW based on interval width
    interval_width_hours: float


class QuantileTimePredictor:
    """
    Predicts remaining time with confidence intervals using quantile regression.
    
    Instead of just predicting mean, we predict 10th, 50th, and 90th percentiles.
    This gives us honest uncertainty estimates.
    """
    
    def __init__(self):
        self.model_median = None   # 50th percentile
        self.model_lower = None    # 10th percentile
        self.model_upper = None    # 90th percentile
        self.median_time = 1.0     # For normalizing interval width
        self.is_fitted = False
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Train three quantile regression models.
        
        Args:
            X: Feature matrix
            y: Target values (remaining time in seconds)
        """
        base_params = {
            'n_estimators': 200,
            'max_depth': 6,
            'learning_rate': 0.1,
            'random_state': 42,
            'verbosity': 0,
        }
        
        # Median (50th percentile) - our point estimate
        self.model_median = xgb.XGBRegressor(
            objective='reg:quantileerror',
            quantile_alpha=0.5,
            **base_params
        )
        self.model_median.fit(X, y)
        
        # Lower bound (10th percentile)
        self.model_lower = xgb.XGBRegressor(
            objective='reg:quantileerror',
            quantile_alpha=0.1,
            **base_params
        )
        self.model_lower.fit(X, y)
        
        # Upper bound (90th percentile)
        self.model_upper = xgb.XGBRegressor(
            objective='reg:quantileerror',
            quantile_alpha=0.9,
            **base_params
        )
        self.model_upper.fit(X, y)
        
        # Store median for relative interval calculation
        self.median_time = np.median(y) if len(y) > 0 else 1.0
        self.is_fitted = True
    
    def predict(self, X: np.ndarray) -> List[TimePredictionWithCI]:
        """
        Predict with confidence intervals.
        
        Args:
            X: Feature matrix
            
        Returns:
            List of TimePredictionWithCI objects
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        
        median = self.model_median.predict(X)
        lower = self.model_lower.predict(X)
        upper = self.model_upper.predict(X)
        
        # Ensure ordering (quantile crossing can happen)
        lower = np.minimum(lower, median)
        upper = np.maximum(upper, median)
        
        # Ensure non-negative
        lower = np.maximum(lower, 0)
        median = np.maximum(median, 0)
        upper = np.maximum(upper, 0)
        
        results = []
        for i in range(len(X)):
            interval_width = (upper[i] - lower[i]) / 3600  # Convert to hours
            
            # Confidence based on relative interval width
            median_hours = self.median_time / 3600 if self.median_time > 0 else 1
            relative_width = interval_width / (median_hours + 1)
            
            if relative_width < 0.5:
                confidence = 'HIGH'
            elif relative_width < 1.0:
                confidence = 'MEDIUM'
            else:
                confidence = 'LOW'
            
            results.append(TimePredictionWithCI(
                point_estimate=median[i] / 3600,
                lower_bound=max(0, lower[i] / 3600),
                upper_bound=upper[i] / 3600,
                confidence_level=confidence,
                interval_width_hours=interval_width
            ))
        
        return results
    
    def predict_single(self, X: np.ndarray) -> TimePredictionWithCI:
        """Predict a single sample."""
        return self.predict(X.reshape(1, -1))[0]
    
    def coverage_score(self, X: np.ndarray, y_true: np.ndarray) -> float:
        """
        Calculate what fraction of true values fall within predicted intervals.
        Target: 80% (since we use 10th-90th percentile).
        """
        predictions = self.predict(X)
        
        covered = 0
        for i, pred in enumerate(predictions):
            true_hours = y_true[i] / 3600
            if pred.lower_bound <= true_hours <= pred.upper_bound:
                covered += 1
        
        return covered / len(predictions) if predictions else 0
    
    def get_state(self) -> dict:
        """Get state for serialization."""
        return {
            'median_time': self.median_time,
            'is_fitted': self.is_fitted,
        }
    
    def set_state(self, state: dict):
        """Set state from deserialized data."""
        self.median_time = state['median_time']
        self.is_fitted = state['is_fitted']


@dataclass
class AggregateConfidence:
    """Aggregate confidence across all predictions."""
    overall_score: float           # 0-1, weighted combination
    overall_level: str             # HIGH, MEDIUM, LOW
    
    next_activity_confidence: float
    next_activity_level: str
    
    outcome_confidence: float
    outcome_level: str
    
    time_confidence: str           # Based on interval width
    
    flags: List[str]               # Specific concerns


def calculate_aggregate_confidence(
    next_activity_prob: float,
    outcome_prob: float,
    time_interval_width: float,
    median_process_time: float
) -> AggregateConfidence:
    """
    Calculate overall prediction confidence.
    
    Weights:
    - Next activity: 40% (most important for immediate action)
    - Outcome: 35% (important for prioritization)
    - Time: 25% (helpful but less critical)
    """
    
    # Normalize time confidence to 0-1 scale
    relative_interval = time_interval_width / (median_process_time + 1)
    time_score = max(0, 1 - relative_interval)  # Narrower = better
    
    # Weighted combination
    overall = (
        0.40 * next_activity_prob +
        0.35 * outcome_prob +
        0.25 * time_score
    )
    
    # Classify each component
    def classify(score: float) -> str:
        if score >= 0.75:
            return 'HIGH'
        if score >= 0.50:
            return 'MEDIUM'
        return 'LOW'
    
    # Identify specific concerns
    flags = []
    if next_activity_prob < 0.5:
        flags.append("Low confidence in next activity prediction")
    if outcome_prob < 0.5:
        flags.append("Outcome prediction is uncertain")
    if time_score < 0.5:
        flags.append("Wide time prediction interval")
    
    return AggregateConfidence(
        overall_score=overall,
        overall_level=classify(overall),
        next_activity_confidence=next_activity_prob,
        next_activity_level=classify(next_activity_prob),
        outcome_confidence=outcome_prob,
        outcome_level=classify(outcome_prob),
        time_confidence=classify(time_score),
        flags=flags
    )
