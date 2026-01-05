"""
Prediction API endpoints.
"""
from typing import List
from fastapi import APIRouter, HTTPException
from datetime import datetime

from app.core.xes_parser import Event
from app.core.predictor import PredictionEngine
from app.core.recommendations import generate_recommendations
from app.api.train import load_predictor
from app.api.split import load_split_data
from app.models.schemas import (
    PredictionRequest, PredictionResult, EventSchema,
    AlternativeActivity, FeatureImportance, TestCase
)
from app.database import get_model

router = APIRouter()


@router.post("/predict/{model_id}", response_model=PredictionResult)
async def predict(model_id: str, request: PredictionRequest):
    """
    Generate prediction for a running case.
    
    Returns WHAT/WHEN/HOW/WHY predictions.
    """
    if not request.events:
        raise HTTPException(status_code=400, detail="At least one event is required")
    
    # Load model
    try:
        predictor = load_predictor(model_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    # Convert request events to Event objects
    events = []
    for ev in request.events:
        events.append(Event(
            activity=ev.activity,
            timestamp=ev.timestamp,
            resource=ev.resource,
            attributes=ev.attributes
        ))
    
    # Create prediction engine and predict
    engine = PredictionEngine(predictor)
    result = engine.predict(events)
    
    # Generate LLM recommendations
    recommendations = generate_recommendations(
        current_activity=events[-1].activity,
        predicted_next=result.next_activity,
        next_probability=result.next_activity_calibrated,
        predicted_outcome=result.predicted_outcome,
        outcome_probability=result.outcome_calibrated,
        remaining_time_hours=result.expected_remaining_time_hours,
        risk_factors=result.risk_factors,
        prefix_length=len(events)
    )
    
    return PredictionResult(
        next_activity=result.next_activity,
        next_activity_probability=result.next_activity_probability,
        next_activity_calibrated=result.next_activity_calibrated,
        next_activity_confidence=result.next_activity_confidence,
        alternatives=[
            AlternativeActivity(activity=a, probability=p)
            for a, p in result.alternative_activities
        ],
        remaining_time_hours=result.expected_remaining_time_hours,
        time_lower_bound_hours=result.time_lower_bound_hours,
        time_upper_bound_hours=result.time_upper_bound_hours,
        time_confidence=result.time_confidence,
        predicted_outcome=result.predicted_outcome,
        outcome_probability=result.outcome_probability,
        outcome_calibrated=result.outcome_calibrated,
        outcome_confidence=result.outcome_confidence,
        outcome_distribution=result.outcome_distribution,
        top_features=[
            FeatureImportance(feature=f, importance=i)
            for f, i in result.feature_importance
        ],
        risk_factors=result.risk_factors,
        aggregate_confidence_score=result.aggregate_score,
        aggregate_confidence_level=result.aggregate_level,
        confidence_flags=result.confidence_flags,
        recommendations=recommendations
    )


@router.get("/test-cases/{log_id}", response_model=List[TestCase])
async def get_test_cases(log_id: str, limit: int = 10):
    """Get sample test cases for demo predictions."""
    from app.database import get_latest_split_for_log
    
    split_info = get_latest_split_for_log(log_id)
    if not split_info:
        raise HTTPException(status_code=404, detail="No split found for this log")
    
    split_id = split_info['split_id']
    
    try:
        split_data = load_split_data(split_id)
        test_traces = split_data['test_traces']
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Split data not found")
    
    cases = []
    for trace in test_traces[:limit]:
        if not trace.events:
            continue
        
        cases.append(TestCase(
            case_id=trace.case_id,
            event_count=len(trace.events),
            final_outcome=trace.events[-1].activity,
            duration_hours=trace.duration_seconds / 3600
        ))
    
    return cases


@router.get("/test-cases/{log_id}/{case_id}/events")
async def get_case_events(log_id: str, case_id: str):
    """Get all events for a specific test case."""
    from app.database import get_latest_split_for_log
    
    split_info = get_latest_split_for_log(log_id)
    if not split_info:
        raise HTTPException(status_code=404, detail="No split found for this log")
    
    split_id = split_info['split_id']
    
    try:
        split_data = load_split_data(split_id)
        test_traces = split_data['test_traces']
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Split data not found")
    
    # Find the trace
    for trace in test_traces:
        if trace.case_id == case_id:
            return {
                "case_id": trace.case_id,
                "events": [
                    {
                        "activity": e.activity,
                        "timestamp": e.timestamp.isoformat(),
                        "resource": e.resource
                    }
                    for e in trace.events
                ]
            }
    
    raise HTTPException(status_code=404, detail="Case not found")
