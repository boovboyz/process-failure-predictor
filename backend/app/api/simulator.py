"""
Simulator API endpoints.
"""
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.core.xes_parser import Trace, Event
from app.core.predictor import PredictionEngine
from app.core.simulator import EventSimulator
from app.api.train import load_predictor
from app.api.split import load_split_data
from app.models.schemas import (
    TraceSummary, LoadTraceRequest, SimulatorState, 
    SimulatorStepResult as SimStepResultSchema, EvolutionPoint
)
from app.database import get_model, save_simulator_session, get_simulator_session
import uuid

router = APIRouter()

# In-memory simulator instances (in production, use Redis or similar)
_simulators: Dict[str, EventSimulator] = {}


def _get_simulator(model_id: str) -> EventSimulator:
    """Get or create simulator for a model."""
    if model_id not in _simulators:
        predictor = load_predictor(model_id)
        engine = PredictionEngine(predictor)
        _simulators[model_id] = EventSimulator(engine)
    return _simulators[model_id]


@router.get("/simulator/{model_id}/traces", response_model=List[TraceSummary])
async def get_available_traces(model_id: str, limit: int = 20):
    """Get list of test traces available for simulation."""
    model_info = get_model(model_id)
    if not model_info:
        raise HTTPException(status_code=404, detail="Model not found")
    
    split_id = model_info.get('split_id')
    if not split_id:
        raise HTTPException(status_code=400, detail="No split associated with model")
    
    try:
        split_data = load_split_data(split_id)
        test_traces = split_data['test_traces']
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Split data not found")
    
    traces = []
    for i, trace in enumerate(test_traces[:limit]):
        if not trace.events:
            continue
        
        traces.append(TraceSummary(
            trace_id=f"{i}",  # Use index as ID
            case_id=trace.case_id,
            event_count=len(trace.events),
            final_activity=trace.events[-1].activity,
            duration_hours=trace.duration_seconds / 3600
        ))
    
    return traces


@router.post("/simulator/{model_id}/load")
async def load_trace(model_id: str, request: LoadTraceRequest):
    """
    Load a trace into the simulator.
    
    Can load by trace_id from test set or provide custom events.
    """
    try:
        simulator = _get_simulator(model_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    if request.events:
        # Custom events provided
        events = [
            Event(
                activity=e.activity,
                timestamp=e.timestamp,
                resource=e.resource,
                attributes=e.attributes
            )
            for e in request.events
        ]
        trace = Trace(case_id="custom", events=events)
    elif request.trace_id is not None:
        # Load from test set
        model_info = get_model(model_id)
        split_id = model_info.get('split_id')
        
        try:
            split_data = load_split_data(split_id)
            test_traces = split_data['test_traces']
            
            trace_idx = int(request.trace_id)
            if trace_idx < 0 or trace_idx >= len(test_traces):
                raise HTTPException(status_code=404, detail="Trace not found")
            
            trace = test_traces[trace_idx]
        except (ValueError, FileNotFoundError):
            raise HTTPException(status_code=404, detail="Trace not found")
    else:
        raise HTTPException(
            status_code=400,
            detail="Either trace_id or events must be provided"
        )
    
    # Load trace into simulator
    state = simulator.load_trace(trace)
    
    # Get initial prediction
    initial_result = simulator.state.prediction_history[0] if simulator.state.prediction_history else None
    
    return {
        "state": state.to_dict(),
        "initial_prediction": initial_result.to_dict() if initial_result else None
    }


@router.post("/simulator/{model_id}/step")
async def step_forward(model_id: str):
    """Advance simulator by one event."""
    try:
        simulator = _get_simulator(model_id)
    except (FileNotFoundError, ValueError) as e:
        raise HTTPException(status_code=404, detail=str(e))
    
    if simulator.state is None:
        raise HTTPException(status_code=400, detail="No trace loaded. Call /load first.")
    
    result = simulator.step_forward()
    
    if result is None:
        return {"message": "Already at end of trace", "is_complete": True}
    
    return {
        "result": result.to_dict(),
        "state": simulator.state.to_dict()
    }


@router.post("/simulator/{model_id}/step-back")
async def step_backward(model_id: str):
    """Go back one event."""
    try:
        simulator = _get_simulator(model_id)
    except (FileNotFoundError, ValueError) as e:
        raise HTTPException(status_code=404, detail=str(e))
    
    if simulator.state is None:
        raise HTTPException(status_code=400, detail="No trace loaded")
    
    result = simulator.step_backward()
    
    if result is None:
        return {"message": "Already at beginning", "step": 0}
    
    return {
        "result": result.to_dict(),
        "state": simulator.state.to_dict()
    }


@router.post("/simulator/{model_id}/jump/{step}")
async def jump_to_step(model_id: str, step: int):
    """Jump to specific step in trace."""
    try:
        simulator = _get_simulator(model_id)
    except (FileNotFoundError, ValueError) as e:
        raise HTTPException(status_code=404, detail=str(e))
    
    if simulator.state is None:
        raise HTTPException(status_code=400, detail="No trace loaded")
    
    result = simulator.jump_to_step(step)
    
    if result is None:
        raise HTTPException(status_code=400, detail="Invalid step")
    
    return {
        "result": result.to_dict(),
        "state": simulator.state.to_dict()
    }


@router.get("/simulator/{model_id}/evolution")
async def get_evolution(model_id: str):
    """Get prediction evolution data for charting."""
    try:
        simulator = _get_simulator(model_id)
    except (FileNotFoundError, ValueError) as e:
        raise HTTPException(status_code=404, detail=str(e))
    
    if simulator.state is None:
        raise HTTPException(status_code=400, detail="No trace loaded")
    
    return {
        "evolution": simulator.get_prediction_evolution(),
        "summary": simulator.get_summary()
    }


@router.post("/simulator/{model_id}/reset")
async def reset_simulator(model_id: str):
    """Reset simulator to beginning of current trace."""
    try:
        simulator = _get_simulator(model_id)
    except (FileNotFoundError, ValueError) as e:
        raise HTTPException(status_code=404, detail=str(e))
    
    if simulator.state is None:
        raise HTTPException(status_code=400, detail="No trace loaded")
    
    state = simulator.reset()
    initial_result = simulator.state.prediction_history[0] if simulator.state.prediction_history else None
    
    return {
        "state": state.to_dict(),
        "initial_prediction": initial_result.to_dict() if initial_result else None
    }


@router.get("/simulator/{model_id}/state")
async def get_current_state(model_id: str):
    """Get current simulator state."""
    try:
        simulator = _get_simulator(model_id)
    except (FileNotFoundError, ValueError) as e:
        raise HTTPException(status_code=404, detail=str(e))
    
    if simulator.state is None:
        return {"loaded": False}
    
    current_result = None
    if simulator.state.current_step < len(simulator.state.prediction_history):
        current_result = simulator.state.prediction_history[simulator.state.current_step]
    
    return {
        "loaded": True,
        "state": simulator.state.to_dict(),
        "current_prediction": current_result.to_dict() if current_result else None
    }
