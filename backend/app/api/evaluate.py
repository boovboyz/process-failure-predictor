"""
Evaluation API endpoints.
"""
from fastapi import APIRouter, HTTPException

from app.core.evaluator import ModelEvaluator
from app.api.train import load_predictor
from app.api.split import load_split_data
from app.models.schemas import EvaluationResults, ConfusionMatrixResponse, EarlyDetectionResponse
from app.database import get_model, get_evaluation, save_evaluation
import uuid

router = APIRouter()


@router.get("/evaluate/{model_id}")
async def evaluate_model(model_id: str):
    """
    Run evaluation on held-out test set.
    
    Returns comprehensive metrics including accuracy, AUC, MAE, and early detection.
    """
    # Check if we have cached results
    cached = get_evaluation(model_id)
    if cached and cached.get('results'):
        return cached['results']
    
    # Load model
    try:
        predictor = load_predictor(model_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    # Get model info for split_id
    model_info = get_model(model_id)
    split_id = model_info.get('split_id')
    
    if not split_id:
        raise HTTPException(status_code=400, detail="No split associated with model")
    
    # Load test traces
    try:
        split_data = load_split_data(split_id)
        test_traces = split_data['test_traces']
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Split data not found")
    
    if not test_traces:
        raise HTTPException(status_code=400, detail="No test traces available")
    
    # Run evaluation
    evaluator = ModelEvaluator(predictor)
    results = evaluator.evaluate(test_traces)
    
    # Cache results
    evaluation_id = str(uuid.uuid4())
    save_evaluation(evaluation_id, model_id, results.to_dict())
    
    return results.to_dict()


@router.get("/evaluate/{model_id}/confusion-matrix", response_model=ConfusionMatrixResponse)
async def get_confusion_matrix(model_id: str):
    """Get detailed confusion matrix for next activity prediction."""
    try:
        predictor = load_predictor(model_id)
    except (FileNotFoundError, ValueError) as e:
        raise HTTPException(status_code=404, detail=str(e))
    
    model_info = get_model(model_id)
    split_id = model_info.get('split_id')
    
    try:
        split_data = load_split_data(split_id)
        test_traces = split_data['test_traces']
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Split data not found")
    
    evaluator = ModelEvaluator(predictor)
    cm_data = evaluator.get_confusion_matrix_details(test_traces)
    
    return ConfusionMatrixResponse(
        matrix=cm_data['matrix'],
        labels=cm_data['labels'],
        total_predictions=cm_data['total_predictions']
    )


@router.get("/evaluate/{model_id}/early-detection", response_model=EarlyDetectionResponse)
async def get_early_detection(model_id: str):
    """Get accuracy at different prefix completion percentages."""
    # Try to get from cached evaluation
    cached = get_evaluation(model_id)
    if cached and cached.get('results', {}).get('early_detection'):
        return EarlyDetectionResponse(results=cached['results']['early_detection'])
    
    # Otherwise compute
    try:
        predictor = load_predictor(model_id)
    except (FileNotFoundError, ValueError) as e:
        raise HTTPException(status_code=404, detail=str(e))
    
    model_info = get_model(model_id)
    split_id = model_info.get('split_id')
    
    try:
        split_data = load_split_data(split_id)
        test_traces = split_data['test_traces']
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Split data not found")
    
    evaluator = ModelEvaluator(predictor)
    results = evaluator.evaluate(test_traces)
    
    return EarlyDetectionResponse(results=results.early_detection)


@router.get("/evaluate/{model_id}/reliability-diagram")
async def get_reliability_diagram(model_id: str):
    """Get data for reliability diagram (calibration visualization)."""
    from app.core.calibration import ConfidenceMetrics
    from app.core.prefix_generator import generate_prefixes
    import numpy as np
    
    try:
        predictor = load_predictor(model_id)
    except (FileNotFoundError, ValueError) as e:
        raise HTTPException(status_code=404, detail=str(e))
    
    model_info = get_model(model_id)
    split_id = model_info.get('split_id')
    
    try:
        split_data = load_split_data(split_id)
        test_traces = split_data['test_traces']
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Split data not found")
    
    # Generate prefixes and get predictions
    prefixes = generate_prefixes(test_traces)
    X = predictor.feature_engineer.transform_batch(prefixes)
    
    y_true = np.array([
        predictor.label_encoder.get(p.target_activity, 0)
        for p in prefixes
    ])
    
    y_prob = predictor.next_activity_model.predict_proba(X)
    
    # Generate reliability diagram data
    diagram_data = ConfidenceMetrics.reliability_diagram_data(y_true, y_prob)
    
    return diagram_data
