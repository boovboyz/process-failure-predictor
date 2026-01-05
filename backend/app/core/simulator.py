"""
Real-time event simulator for step-by-step prediction visualization.
"""
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from enum import Enum

from app.core.xes_parser import Trace, Event
from app.core.predictor import PredictionEngine, PredictionResult
from app.core.recommendations import generate_recommendations


class SimulatorMode(Enum):
    MANUAL = "manual"
    AUTO_PLAY = "auto_play"


@dataclass
class SimulatorState:
    """Current state of the event simulator."""
    trace: Trace
    current_step: int  # 0-indexed, how many events we've "seen"
    total_steps: int
    prediction_history: List[Dict] = field(default_factory=list)
    
    @property
    def current_prefix(self) -> List[Event]:
        """Get events seen so far."""
        return self.trace.events[:self.current_step + 1]
    
    @property
    def actual_next_event(self) -> Optional[Event]:
        """Get the actual next event (ground truth)."""
        if self.current_step + 1 < len(self.trace.events):
            return self.trace.events[self.current_step + 1]
        return None
    
    @property
    def is_complete(self) -> bool:
        """Check if we've reached the end of the trace."""
        return self.current_step >= len(self.trace.events) - 1
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "trace_id": self.trace.case_id,
            "current_step": self.current_step,
            "total_steps": self.total_steps,
            "is_complete": self.is_complete,
        }


@dataclass
class SimulatorStepResult:
    """Result of a simulator step."""
    step: int
    total_steps: int
    
    # Current state
    current_events: List[Dict[str, Any]]
    current_activity: str
    
    # Prediction
    predicted_next: str
    predicted_probability: float
    predicted_confidence: str
    alternatives: List[Dict[str, float]]
    predicted_time_hours: float
    time_lower_hours: float
    time_upper_hours: float
    predicted_outcome: str
    outcome_probability: float
    outcome_confidence: str
    risk_factors: List[str]
    
    # Ground truth (if available)
    actual_next: Optional[str]
    actual_time_hours: Optional[float]
    actual_outcome: Optional[str]
    
    # Comparison
    next_correct: Optional[bool]
    time_error_hours: Optional[float]
    outcome_correct: Optional[bool]
    
    # Recommendations
    recommendations: Optional[List[str]]
    
    def to_dict(self) -> Dict[str, Any]:
        # Helper to convert numpy types to Python native types
        def to_native(val):
            if val is None:
                return None
            if isinstance(val, (list, tuple)):
                return [to_native(v) for v in val]
            if isinstance(val, dict):
                return {k: to_native(v) for k, v in val.items()}
            if hasattr(val, 'item'):  # numpy scalar
                return val.item()
            return val
        
        return {
            "step": int(self.step),
            "total_steps": int(self.total_steps),
            "current_events": self.current_events,
            "current_activity": str(self.current_activity),
            "predicted_next": str(self.predicted_next),
            "predicted_probability": float(self.predicted_probability),
            "predicted_confidence": str(self.predicted_confidence),
            "alternatives": [{"activity": str(a.get("activity", a.get("activity", ""))), "probability": float(a.get("probability", 0))} for a in self.alternatives],
            "predicted_time_hours": float(self.predicted_time_hours),
            "time_lower_hours": float(self.time_lower_hours),
            "time_upper_hours": float(self.time_upper_hours),
            "predicted_outcome": str(self.predicted_outcome),
            "outcome_probability": float(self.outcome_probability),
            "outcome_confidence": str(self.outcome_confidence),
            "risk_factors": [str(r) for r in self.risk_factors],
            "actual_next": str(self.actual_next) if self.actual_next else None,
            "actual_time_hours": float(self.actual_time_hours) if self.actual_time_hours is not None else None,
            "actual_outcome": str(self.actual_outcome) if self.actual_outcome else None,
            "next_correct": bool(self.next_correct) if self.next_correct is not None else None,
            "time_error_hours": float(self.time_error_hours) if self.time_error_hours is not None else None,
            "outcome_correct": bool(self.outcome_correct) if self.outcome_correct is not None else None,
            "recommendations": [str(r) for r in self.recommendations] if self.recommendations else None,
        }


class EventSimulator:
    """
    Simulates stepping through a trace event-by-event.
    
    At each step, generates predictions and compares with ground truth.
    """
    
    def __init__(self, prediction_engine: PredictionEngine):
        self.prediction_engine = prediction_engine
        self.state: Optional[SimulatorState] = None
    
    def load_trace(self, trace: Trace) -> SimulatorState:
        """
        Initialize simulator with a trace.
        
        Args:
            trace: Trace to simulate
            
        Returns:
            Initial SimulatorState
        """
        self.state = SimulatorState(
            trace=trace,
            current_step=0,
            total_steps=len(trace.events),
            prediction_history=[]
        )
        
        # Generate initial prediction
        self._generate_prediction()
        
        return self.state
    
    def step_forward(self) -> Optional[SimulatorStepResult]:
        """
        Advance to next event and generate new prediction.
        
        Returns:
            SimulatorStepResult or None if already at end
        """
        if self.state is None or self.state.is_complete:
            return None
        
        self.state.current_step += 1
        return self._generate_prediction()
    
    def step_backward(self) -> Optional[SimulatorStepResult]:
        """
        Go back to previous event.
        
        Returns:
            SimulatorStepResult or None if at beginning
        """
        if self.state is None or self.state.current_step <= 0:
            return None
        
        self.state.current_step -= 1
        
        # Return cached prediction if available
        if self.state.current_step < len(self.state.prediction_history):
            return self.state.prediction_history[self.state.current_step]
        
        return self._generate_prediction()
    
    def jump_to_step(self, step: int) -> Optional[SimulatorStepResult]:
        """
        Jump to specific step in trace.
        
        Args:
            step: Step number to jump to
            
        Returns:
            SimulatorStepResult at that step
        """
        if self.state is None:
            return None
        
        step = max(0, min(step, self.state.total_steps - 1))
        self.state.current_step = step
        
        # Return cached prediction if available
        if step < len(self.state.prediction_history):
            return self.state.prediction_history[step]
        
        return self._generate_prediction()
    
    def reset(self) -> Optional[SimulatorState]:
        """Reset simulator to beginning of current trace."""
        if self.state is None:
            return None
        
        self.state.current_step = 0
        self.state.prediction_history = []
        self._generate_prediction()
        
        return self.state
    
    def _generate_prediction(self) -> SimulatorStepResult:
        """Generate prediction for current prefix."""
        prefix_events = self.state.current_prefix
        
        # Get prediction from engine
        prediction = self.prediction_engine.predict(prefix_events)
        
        # Get ground truth
        actual_next = self.state.actual_next_event
        actual_outcome = self.state.trace.events[-1].activity
        
        # Calculate actual remaining time
        if prefix_events:
            actual_remaining = (
                self.state.trace.end_time - prefix_events[-1].timestamp
            ).total_seconds() / 3600
        else:
            actual_remaining = 0
        
        # Build result
        result = SimulatorStepResult(
            step=self.state.current_step,
            total_steps=self.state.total_steps,
            current_events=[
                {
                    "activity": e.activity,
                    "timestamp": e.timestamp.isoformat(),
                    "resource": e.resource
                }
                for e in prefix_events
            ],
            current_activity=prefix_events[-1].activity if prefix_events else "",
            predicted_next=prediction.next_activity,
            predicted_probability=prediction.next_activity_calibrated,
            predicted_confidence=prediction.next_activity_confidence,
            alternatives=[
                {"activity": a, "probability": p}
                for a, p in prediction.alternative_activities
            ],
            predicted_time_hours=prediction.expected_remaining_time_hours,
            time_lower_hours=prediction.time_lower_bound_hours,
            time_upper_hours=prediction.time_upper_bound_hours,
            predicted_outcome=prediction.predicted_outcome,
            outcome_probability=prediction.outcome_calibrated,
            outcome_confidence=prediction.outcome_confidence,
            risk_factors=prediction.risk_factors,
            actual_next=actual_next.activity if actual_next else None,
            actual_time_hours=actual_remaining if actual_next else None,
            actual_outcome=actual_outcome,
            next_correct=(
                prediction.next_activity == actual_next.activity
                if actual_next else None
            ),
            time_error_hours=(
                abs(prediction.expected_remaining_time_hours - actual_remaining)
                if actual_next else None
            ),
            outcome_correct=prediction.predicted_outcome == actual_outcome,
            recommendations=generate_recommendations(
                current_activity=prefix_events[-1].activity if prefix_events else "",
                predicted_next=prediction.next_activity,
                next_probability=prediction.next_activity_calibrated,
                predicted_outcome=prediction.predicted_outcome,
                outcome_probability=prediction.outcome_calibrated,
                remaining_time_hours=prediction.expected_remaining_time_hours,
                risk_factors=prediction.risk_factors,
                prefix_length=len(prefix_events)
            )
        )
        
        # Cache result
        if self.state.current_step >= len(self.state.prediction_history):
            self.state.prediction_history.append(result)
        else:
            self.state.prediction_history[self.state.current_step] = result
        
        return result
    
    def get_prediction_evolution(self) -> List[Dict[str, Any]]:
        """
        Get how predictions evolved over all steps.
        
        Useful for charting.
        """
        if self.state is None:
            return []
        
        evolution = []
        for i, result in enumerate(self.state.prediction_history):
            evolution.append({
                "step": i + 1,
                "prefix_length": i + 1,
                "completion_pct": (i + 1) / self.state.total_steps * 100,
                "next_activity_confidence": result.predicted_probability,
                "next_correct": result.next_correct,
                "outcome_confidence": result.outcome_probability,
                "outcome_correct": result.outcome_correct,
                "time_error": result.time_error_hours
            })
        
        return evolution
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics for the simulation."""
        if not self.state or not self.state.prediction_history:
            return {}
        
        history = self.state.prediction_history
        
        # Count correct predictions
        next_correct = sum(1 for r in history if r.next_correct is True)
        next_total = sum(1 for r in history if r.next_correct is not None)
        
        outcome_correct = sum(1 for r in history if r.outcome_correct is True)
        
        time_errors = [r.time_error_hours for r in history if r.time_error_hours is not None]
        
        return {
            "total_steps": len(history),
            "next_activity_accuracy": next_correct / next_total if next_total > 0 else 0,
            "outcome_correct": outcome_correct > 0,
            "avg_time_error_hours": sum(time_errors) / len(time_errors) if time_errors else 0,
            "final_prediction_correct": history[-1].next_correct if history else None,
        }
