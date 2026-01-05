"""
Upload API endpoints.
"""
import uuid
import os
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException
import tempfile

from app.core.xes_parser import parse_xes, validate_log
from app.models.schemas import UploadResponse
from app.database import save_event_log

router = APIRouter()

# Directory for storing uploaded files
UPLOAD_DIR = Path(__file__).parent.parent.parent / "data" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@router.post("/upload", response_model=UploadResponse)
async def upload_xes(file: UploadFile = File(...)):
    """
    Upload and parse XES file.
    
    Returns statistics about the uploaded event log.
    """
    if not file.filename.endswith('.xes'):
        raise HTTPException(status_code=400, detail="File must be a .xes file")
    
    # Generate unique ID
    log_id = str(uuid.uuid4())
    
    # Save file temporarily for parsing
    file_path = UPLOAD_DIR / f"{log_id}.xes"
    
    try:
        # Save uploaded file
        content = await file.read()
        with open(file_path, 'wb') as f:
            f.write(content)
        
        # Parse XES file
        event_log = parse_xes(str(file_path))
        
        # Validate
        issues = validate_log(event_log)
        if issues:
            # Don't fail, just log warnings
            print(f"Validation warnings for {log_id}: {issues}")
        
        if not event_log.traces:
            raise HTTPException(status_code=400, detail="No valid traces found in XES file")
        
        # Get time range
        time_range = event_log.time_range
        if time_range[0] is None:
            raise HTTPException(status_code=400, detail="Could not determine time range")
        
        # Save metadata to database
        save_event_log(
            log_id=log_id,
            filename=file.filename,
            trace_count=len(event_log.traces),
            event_count=event_log.total_events,
            unique_activities=len(event_log.all_activities),
            time_range_start=time_range[0],
            time_range_end=time_range[1],
            activities=event_log.all_activities[:20]  # Store sample
        )
        
        return UploadResponse(
            log_id=log_id,
            trace_count=len(event_log.traces),
            event_count=event_log.total_events,
            unique_activities=len(event_log.all_activities),
            time_range_start=time_range[0],
            time_range_end=time_range[1],
            sample_activities=event_log.all_activities[:10]
        )
        
    except ValueError as e:
        # Clean up file if parsing fails
        if file_path.exists():
            file_path.unlink()
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        if file_path.exists():
            file_path.unlink()
        raise HTTPException(status_code=500, detail=f"Failed to process file: {str(e)}")


@router.get("/logs/{log_id}")
async def get_log_info(log_id: str):
    """Get information about an uploaded log."""
    from app.database import get_event_log
    
    log_info = get_event_log(log_id)
    if not log_info:
        raise HTTPException(status_code=404, detail="Log not found")
    
    return log_info


@router.get("/logs/{log_id}/preview")
async def get_log_preview(log_id: str, limit: int = 5):
    """Get a preview of traces from an uploaded log."""
    file_path = UPLOAD_DIR / f"{log_id}.xes"
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Log file not found")
    
    event_log = parse_xes(str(file_path))
    
    preview = []
    for trace in event_log.traces[:limit]:
        preview.append({
            "case_id": trace.case_id,
            "event_count": len(trace.events),
            "start_time": trace.start_time.isoformat() if trace.events else None,
            "end_time": trace.end_time.isoformat() if trace.events else None,
            "activities": [e.activity for e in trace.events[:10]]
        })
    
    return {"traces": preview}
