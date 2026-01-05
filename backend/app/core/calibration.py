"""
Probability calibration for prediction confidence.
Uses isotonic regression for well-calibrated probabilities.
"""
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import calibration_curve
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional


@dataclass
class CalibrationResult:
    """Result of calibrated prediction."""
    calibrated_probs: np.ndarray
    confidence_level: str  # HIGH, MEDIUM, LOW
    reliability_score: float


class ProbabilityCalibrator:
    """
    Calibrates model probabilities using isotonic regression.
    
    Why isotonic over Platt scaling?
    - Non-parametric: doesn't assume sigmoid shape
    - Better for XGBoost which has complex probability distributions
    - Monotonic: preserves ranking of predictions
    """
    
    def __init__(self):
        self.calibrators: Dict[int, IsotonicRegression] = {}
        self.confidence_thresholds = {
            'HIGH': 0.75,
            'MEDIUM': 0.50,
            'LOW': 0.0
        }
        self.n_classes = 0
        self.is_fitted = False
    
    def fit(self, y_true: np.ndarray, y_prob: np.ndarray):
        """
        Fit calibrators on validation set predictions.
        
        Args:
            y_true: True labels (n_samples,)
            y_prob: Predicted probabilities (n_samples, n_classes)
        """
        if len(y_prob.shape) == 1:
            # Binary case
            y_prob = np.column_stack([1 - y_prob, y_prob])
        
        self.n_classes = y_prob.shape[1]
        
        for class_idx in range(self.n_classes):
            # Binary indicator for this class
            y_binary = (y_true == class_idx).astype(int)
            prob_class = y_prob[:, class_idx]
            
            # Fit isotonic regression
            calibrator = IsotonicRegression(out_of_bounds='clip')
            
            # Need at least 2 unique values
            if len(np.unique(prob_class)) >= 2:
                calibrator.fit(prob_class, y_binary)
                self.calibrators[class_idx] = calibrator
        
        self.is_fitted = True
    
    def calibrate(self, y_prob: np.ndarray) -> np.ndarray:
        """
        Apply calibration to raw probabilities.
        
        Args:
            y_prob: Raw probabilities (n_samples, n_classes) or (n_samples,)
            
        Returns:
            Calibrated probabilities (n_samples, n_classes)
        """
        if not self.is_fitted:
            return y_prob
        
        if len(y_prob.shape) == 1:
            y_prob = np.column_stack([1 - y_prob, y_prob])
        
        calibrated = np.zeros_like(y_prob)
        
        for class_idx in range(y_prob.shape[1]):
            if class_idx in self.calibrators:
                calibrated[:, class_idx] = self.calibrators[class_idx].predict(
                    y_prob[:, class_idx]
                )
            else:
                calibrated[:, class_idx] = y_prob[:, class_idx]
        
        # Renormalize to sum to 1
        row_sums = calibrated.sum(axis=1, keepdims=True)
        calibrated = calibrated / (row_sums + 1e-10)
        
        return calibrated
    
    def calibrate_single(self, y_prob: np.ndarray) -> np.ndarray:
        """Calibrate a single prediction."""
        return self.calibrate(y_prob.reshape(1, -1))[0]
    
    def get_confidence_level(self, prob: float) -> str:
        """Classify probability into confidence level."""
        if prob >= self.confidence_thresholds['HIGH']:
            return 'HIGH'
        elif prob >= self.confidence_thresholds['MEDIUM']:
            return 'MEDIUM'
        else:
            return 'LOW'
    
    def predict_with_confidence(self, y_prob: np.ndarray) -> List[CalibrationResult]:
        """Get calibrated predictions with confidence levels."""
        calibrated = self.calibrate(y_prob)
        results = []
        
        for i in range(len(calibrated)):
            max_prob = calibrated[i].max()
            results.append(CalibrationResult(
                calibrated_probs=calibrated[i],
                confidence_level=self.get_confidence_level(max_prob),
                reliability_score=max_prob
            ))
        
        return results


class ConfidenceMetrics:
    """Calculate calibration quality metrics."""
    
    @staticmethod
    def expected_calibration_error(y_true: np.ndarray,
                                   y_prob: np.ndarray,
                                   n_bins: int = 10) -> float:
        """
        Expected Calibration Error (ECE).
        
        Lower is better. <0.05 is well-calibrated.
        
        ECE = Σ (|bin_size| / n) * |accuracy(bin) - confidence(bin)|
        """
        if len(y_prob.shape) == 1:
            y_prob = np.column_stack([1 - y_prob, y_prob])
        
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        
        y_pred = y_prob.argmax(axis=1)
        confidences = y_prob.max(axis=1)
        accuracies = (y_pred == y_true).astype(float)
        
        for i in range(n_bins):
            in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                avg_confidence = confidences[in_bin].mean()
                avg_accuracy = accuracies[in_bin].mean()
                ece += prop_in_bin * abs(avg_accuracy - avg_confidence)
        
        return ece
    
    @staticmethod
    def reliability_diagram_data(y_true: np.ndarray,
                                 y_prob: np.ndarray,
                                 n_bins: int = 10) -> Dict:
        """Generate data for reliability diagram visualization."""
        if len(y_prob.shape) == 1:
            y_prob = np.column_stack([1 - y_prob, y_prob])
        
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        
        y_pred = y_prob.argmax(axis=1)
        confidences = y_prob.max(axis=1)
        accuracies = (y_pred == y_true).astype(float)
        
        bins = []
        for i in range(n_bins):
            in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
            
            if in_bin.sum() > 0:
                bins.append({
                    'bin_center': (bin_boundaries[i] + bin_boundaries[i + 1]) / 2,
                    'avg_confidence': float(confidences[in_bin].mean()),
                    'avg_accuracy': float(accuracies[in_bin].mean()),
                    'count': int(in_bin.sum()),
                    'gap': float(abs(accuracies[in_bin].mean() - confidences[in_bin].mean()))
                })
        
        ece = ConfidenceMetrics.expected_calibration_error(y_true, y_prob, n_bins)
        
        return {
            'bins': bins,
            'ece': ece,
            'perfect_calibration': [{'x': i/10, 'y': i/10} for i in range(11)]
        }
    
    @staticmethod
    def confidence_distribution(y_prob: np.ndarray) -> Dict[str, float]:
        """Calculate distribution of predictions across confidence levels."""
        if len(y_prob.shape) == 1:
            y_prob = np.column_stack([1 - y_prob, y_prob])
        
        max_probs = y_prob.max(axis=1)
        
        high = (max_probs >= 0.75).mean()
        medium = ((max_probs >= 0.50) & (max_probs < 0.75)).mean()
        low = (max_probs < 0.50).mean()
        
        return {
            'HIGH': high,
            'MEDIUM': medium,
            'LOW': low
        }
