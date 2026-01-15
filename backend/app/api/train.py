"""
Training API endpoints.
"""
import uuid
import time
from pathlib import Path
from typing import Optional
from fastapi import APIRouter, HTTPException, BackgroundTasks

from app.core.trainer import ProcessPredictor
from app.api.split import load_split_data
from app.models.schemas import TrainingResponse, ModelStatus, TrainingMetrics, TrainingRequest
from app.database import (
    get_latest_split_for_log, save_model, update_model_status, get_model
)

router = APIRouter()

MODEL_DIR = Path(__file__).parent.parent.parent / "data" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def _train_model_async(model_id: str, split_id: str, model_path: str):
    """Background task for model training."""
    try:
        # Load split data
        split_data = load_split_data(split_id)
        train_traces = split_data['train_traces']
        
        # Train model
        predictor = ProcessPredictor()
        metrics = predictor.train(train_traces)
        
        # Save model
        predictor.save(model_path)
        
        # Update status
        update_model_status(
            model_id,
            status="ready",
            training_time_seconds=metrics.get('training_time_seconds', 0),
            metrics=metrics,
            model_path=model_path
        )
        
    except Exception as e:
        update_model_status(model_id, status="failed", metrics={"error": str(e)})


@router.post("/train/{log_id}", response_model=TrainingResponse)
async def train_models(
    log_id: str, 
    background_tasks: BackgroundTasks,
    request: TrainingRequest = TrainingRequest()
):
    """
    Train prediction models on split data.
    
    Training runs synchronously. Use /models/{model_id}/status to check progress.
    
    Parameters:
    - model_type: "xgboost" (default) or "lightgbm"
    - auto_tune: Enable Optuna hyperparameter tuning (slower but better)
    - tune_trials: Number of Optuna trials (10-200)
    - tune_timeout: Timeout in seconds for tuning
    """
    # Validate model_type
    if request.model_type not in ("xgboost", "lightgbm"):
        raise HTTPException(
            status_code=400,
            detail="model_type must be 'xgboost' or 'lightgbm'"
        )
    
    # Get latest split
    split_info = get_latest_split_for_log(log_id)
    if not split_info:
        raise HTTPException(
            status_code=400,
            detail="No split found. Please run /split/{log_id} first."
        )
    
    split_id = split_info['split_id']
    
    # Generate model ID
    model_id = str(uuid.uuid4())
    model_path = str(MODEL_DIR / f"{model_id}.pkl")
    
    # Save initial status
    save_model(
        model_id=model_id,
        log_id=log_id,
        split_id=split_id,
        status="training",
        model_path=model_path
    )
    
    try:
        start_time = time.time()
        
        split_data = load_split_data(split_id)
        train_traces = split_data['train_traces']
        
        # Create predictor with specified model type
        predictor = ProcessPredictor(model_type=request.model_type)
        
        # Train with optional hyperparameter tuning
        metrics = predictor.train(
            train_traces,
            auto_tune=request.auto_tune,
            tune_trials=request.tune_trials,
        )
        
        predictor.save(model_path)
        
        training_time = time.time() - start_time
        
        # Add model config to metrics
        metrics['config'] = {
            'model_type': request.model_type,
            'auto_tune': request.auto_tune,
            'tune_trials': request.tune_trials if request.auto_tune else None,
        }
        
        update_model_status(
            model_id,
            status="ready",
            training_time_seconds=training_time,
            metrics=metrics,
            model_path=model_path
        )
        
        return TrainingResponse(
            model_id=model_id,
            training_time_seconds=training_time,
            metrics={
                'next_activity': TrainingMetrics(
                    loss=0.0,
                    validation_score=metrics.get('next_activity', {}).get('accuracy', 0)
                ),
                'outcome': TrainingMetrics(
                    loss=0.0,
                    validation_score=metrics.get('outcome', {}).get('accuracy', 0)
                ),
                'time': TrainingMetrics(
                    loss=metrics.get('time', {}).get('mae_hours', 0),
                    validation_score=metrics.get('time', {}).get('coverage', 0)
                ),
            }
        )
        
    except Exception as e:
        update_model_status(model_id, status="failed")
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")


@router.get("/models/{model_id}/status", response_model=ModelStatus)
async def model_status(model_id: str):
    """Get training status and metrics."""
    model_info = get_model(model_id)
    if not model_info:
        raise HTTPException(status_code=404, detail="Model not found")
    
    return ModelStatus(
        model_id=model_id,
        status=model_info.get('status', 'unknown'),
        progress=1.0 if model_info.get('status') == 'ready' else None,
        error=model_info.get('metrics', {}).get('error') if isinstance(model_info.get('metrics'), dict) else None
    )


def load_predictor(model_id: str) -> ProcessPredictor:
    """Load a trained predictor."""
    model_info = get_model(model_id)
    if not model_info:
        raise FileNotFoundError(f"Model not found: {model_id}")
    
    if model_info.get('status') != 'ready':
        raise ValueError(f"Model not ready: {model_info.get('status')}")
    
    model_path = model_info.get('model_path')
    if not model_path or not Path(model_path).exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    return ProcessPredictor.load(model_path)
