"""
Split API endpoints.
"""
import uuid
from pathlib import Path
from fastapi import APIRouter, HTTPException
import pickle

from app.core.xes_parser import parse_xes
from app.core.splitter import temporal_split, adaptive_split
from app.models.schemas import SplitResponse
from app.database import save_split, get_event_log, get_latest_split_for_log

router = APIRouter()

UPLOAD_DIR = Path(__file__).parent.parent.parent / "data" / "uploads"
SPLIT_DIR = Path(__file__).parent.parent.parent / "data" / "splits"
SPLIT_DIR.mkdir(parents=True, exist_ok=True)


@router.post("/split/{log_id}", response_model=SplitResponse)
async def split_data(log_id: str, train_ratio: float = 0.9):
    """
    Perform temporal split on uploaded log.
    
    Uses trace-aware splitting to prevent data leakage.
    """
    if not 0.1 <= train_ratio <= 0.95:
        raise HTTPException(
            status_code=400,
            detail="train_ratio must be between 0.1 and 0.95"
        )
    
    # Check log exists
    log_info = get_event_log(log_id)
    if not log_info:
        raise HTTPException(status_code=404, detail="Log not found")
    
    # Load event log
    file_path = UPLOAD_DIR / f"{log_id}.xes"
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Log file not found")
    
    event_log = parse_xes(str(file_path))
    
    try:
        # Perform adaptive split (adjusts ratio to reduce exclusions)
        split_result, actual_ratio = adaptive_split(
            event_log,
            target_train_ratio=train_ratio,
            max_exclusion_rate=0.15
        )
        
        # Generate split ID
        split_id = str(uuid.uuid4())
        
        # Save split data for later use
        split_data = {
            'train_traces': split_result.train_traces,
            'test_traces': split_result.test_traces,
            'cutoff_time': split_result.cutoff_time,
        }
        
        split_file = SPLIT_DIR / f"{split_id}.pkl"
        with open(split_file, 'wb') as f:
            pickle.dump(split_data, f)
        
        # Save metadata
        save_split(
            split_id=split_id,
            log_id=log_id,
            train_ratio=actual_ratio,
            train_traces=len(split_result.train_traces),
            test_traces=len(split_result.test_traces),
            excluded_traces=len(split_result.excluded_traces),
            cutoff_time=split_result.cutoff_time
        )
        
        warnings = split_result.get_warnings()
        if actual_ratio != train_ratio:
            warnings.append(
                f"Split ratio adjusted from {train_ratio:.0%} to {actual_ratio:.0%} "
                "to reduce exclusions"
            )
        
        return SplitResponse(
            train_traces=len(split_result.train_traces),
            test_traces=len(split_result.test_traces),
            excluded_traces=len(split_result.excluded_traces),
            cutoff_time=split_result.cutoff_time,
            effective_ratio=split_result.effective_train_ratio,
            warnings=warnings
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/split/{log_id}/latest")
async def get_latest_split(log_id: str):
    """Get the most recent split for a log."""
    split_info = get_latest_split_for_log(log_id)
    if not split_info:
        raise HTTPException(status_code=404, detail="No split found for this log")
    
    return split_info


def load_split_data(split_id: str) -> dict:
    """Load split data from file."""
    split_file = SPLIT_DIR / f"{split_id}.pkl"
    if not split_file.exists():
        raise FileNotFoundError(f"Split file not found: {split_id}")
    
    with open(split_file, 'rb') as f:
        return pickle.load(f)
