"""
Pydantic schemas for the Process Failure Predictor API.
"""
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Any
from pydantic import BaseModel, Field


# ─────────────────────────────────────────────────────────────────────────────
# DATA STRUCTURES
# ─────────────────────────────────────────────────────────────────────────────

class EventSchema(BaseModel):
    """Single event in a process trace."""
    activity: str
    timestamp: datetime
    resource: Optional[str] = None
    attributes: Dict[str, Any] = Field(default_factory=dict)


class TraceSchema(BaseModel):
    """A single process trace (case)."""
    case_id: str
    events: List[EventSchema]
    attributes: Dict[str, Any] = Field(default_factory=dict)


class EventLogSchema(BaseModel):
    """Complete event log."""
    traces: List[TraceSchema]
    attributes: Dict[str, Any] = Field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# UPLOAD RESPONSES
# ─────────────────────────────────────────────────────────────────────────────

class UploadResponse(BaseModel):
    """Response from XES file upload."""
    log_id: str
    trace_count: int
    event_count: int
    unique_activities: int
    time_range_start: datetime
    time_range_end: datetime
    sample_activities: List[str]


# ─────────────────────────────────────────────────────────────────────────────
# SPLIT RESPONSES
# ─────────────────────────────────────────────────────────────────────────────

class SplitResponse(BaseModel):
    """Response from temporal split."""
    train_traces: int
    test_traces: int
    excluded_traces: int
    cutoff_time: datetime
    effective_ratio: float
    warnings: List[str] = Field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING RESPONSES
# ─────────────────────────────────────────────────────────────────────────────

class TrainingMetrics(BaseModel):
    """Metrics from training a single model."""
    loss: float
    validation_score: Optional[float] = None


class TrainingResponse(BaseModel):
    """Response from model training."""
    model_id: str
    training_time_seconds: float
    metrics: Dict[str, TrainingMetrics]


class ModelStatus(BaseModel):
    """Model training status."""
    model_id: str
    status: str  # "training", "ready", "failed"
    progress: Optional[float] = None
    error: Optional[str] = None


# ─────────────────────────────────────────────────────────────────────────────
# EVALUATION RESPONSES
# ─────────────────────────────────────────────────────────────────────────────

class EvaluationResults(BaseModel):
    """Comprehensive evaluation results on test set."""
    # Next Activity
    next_activity_accuracy: float
    next_activity_top3_accuracy: float
    next_activity_macro_f1: float
    confusion_matrix: List[List[int]]
    class_labels: List[str]
    
    # Outcome
    outcome_auc_roc: float
    outcome_precision: float
    outcome_recall: float
    outcome_f1: float
    
    # Time
    time_mae_hours: float
    time_rmse_hours: float
    time_mape: float
    
    # Early Detection
    early_detection: Dict[str, float]  # {"25%": 0.71, "50%": 0.82, "75%": 0.89}
    
    # Calibration
    expected_calibration_error: float
    
    # Samples
    sample_predictions: List[Dict[str, Any]]


class ConfusionMatrixResponse(BaseModel):
    """Detailed confusion matrix."""
    matrix: List[List[int]]
    labels: List[str]
    total_predictions: int


class EarlyDetectionResponse(BaseModel):
    """Accuracy at different prefix completion percentages."""
    results: Dict[str, float]  # {"25%": 0.71, ...}


# ─────────────────────────────────────────────────────────────────────────────
# PREDICTION SCHEMAS
# ─────────────────────────────────────────────────────────────────────────────

class PredictionRequest(BaseModel):
    """Request for prediction on a running case."""
    events: List[EventSchema]


class AlternativeActivity(BaseModel):
    """Alternative activity prediction."""
    activity: str
    probability: float


class FeatureImportance(BaseModel):
    """Feature importance for explanation."""
    feature: str
    importance: float


class PredictionResult(BaseModel):
    """Comprehensive prediction result."""
    # WHAT - Next activity
    next_activity: str
    next_activity_probability: float
    next_activity_calibrated: float
    next_activity_confidence: str  # HIGH, MEDIUM, LOW
    alternatives: List[AlternativeActivity]
    
    # WHEN - Timing
    remaining_time_hours: float
    time_lower_bound_hours: float
    time_upper_bound_hours: float
    time_confidence: str  # HIGH, MEDIUM, LOW
    
    # HOW - Outcome
    predicted_outcome: str
    outcome_probability: float
    outcome_calibrated: float
    outcome_confidence: str  # HIGH, MEDIUM, LOW
    outcome_distribution: Dict[str, float]
    
    # WHY - Explanation
    top_features: List[FeatureImportance]
    risk_factors: List[str]
    
    # Overall Confidence
    aggregate_confidence_score: float
    aggregate_confidence_level: str
    confidence_flags: List[str]
    
    # ACTION (optional)
    recommendations: Optional[List[str]] = None


class TestCase(BaseModel):
    """Summary of a test case for selection."""
    case_id: str
    event_count: int
    final_outcome: str
    duration_hours: float


# ─────────────────────────────────────────────────────────────────────────────
# SIMULATOR SCHEMAS
# ─────────────────────────────────────────────────────────────────────────────

class TraceSummary(BaseModel):
    """Summary of a trace for simulator selection."""
    trace_id: str
    case_id: str
    event_count: int
    final_activity: str
    duration_hours: float


class LoadTraceRequest(BaseModel):
    """Request to load a trace into simulator."""
    trace_id: Optional[str] = None
    events: Optional[List[EventSchema]] = None


class SimulatorState(BaseModel):
    """Current state of the event simulator."""
    trace_id: str
    total_steps: int
    current_step: int
    is_complete: bool


class SimulatorStepResult(BaseModel):
    """Result of a simulator step."""
    step: int
    total_steps: int
    current_events: List[Dict[str, Any]]
    
    # Predictions
    predicted_next: str
    predicted_probability: float
    predicted_confidence: str
    alternatives: List[AlternativeActivity]
    predicted_time_hours: float
    predicted_outcome: str
    outcome_probability: float
    risk_factors: List[str]
    
    # Ground truth
    actual_next: Optional[str] = None
    actual_time_hours: Optional[float] = None
    actual_outcome: Optional[str] = None
    
    # Comparison
    next_correct: Optional[bool] = None
    time_error_hours: Optional[float] = None
    outcome_correct: Optional[bool] = None
    
    recommendations: Optional[List[str]] = None


class EvolutionPoint(BaseModel):
    """Single point in prediction evolution chart."""
    step: int
    completion_pct: float
    next_confidence: float
    next_correct: bool
    outcome_confidence: float
    outcome_correct: bool
    time_error: Optional[float] = None


# ─────────────────────────────────────────────────────────────────────────────
# CALIBRATION SCHEMAS
# ─────────────────────────────────────────────────────────────────────────────

class ReliabilityBin(BaseModel):
    """Single bin in a reliability diagram."""
    bin_center: float
    avg_confidence: float
    avg_accuracy: float
    count: int
    gap: float


class ReliabilityDiagramData(BaseModel):
    """Data for reliability diagram visualization."""
    bins: List[ReliabilityBin]
    ece: float  # Expected Calibration Error
