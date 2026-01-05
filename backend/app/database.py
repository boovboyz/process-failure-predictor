"""
Database configuration and session management using SQLite.
"""
import sqlite3
import json
import os
from datetime import datetime
from typing import Optional, Dict, Any, List
from pathlib import Path

# Database path
DB_DIR = Path(__file__).parent.parent.parent / "data"
DB_PATH = DB_DIR / "process_predictor.db"


def ensure_db_dir():
    """Ensure the data directory exists."""
    DB_DIR.mkdir(parents=True, exist_ok=True)


def get_connection() -> sqlite3.Connection:
    """Get a database connection."""
    ensure_db_dir()
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Initialize database tables."""
    conn = get_connection()
    cursor = conn.cursor()
    
    # Event logs table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS event_logs (
            log_id TEXT PRIMARY KEY,
            filename TEXT NOT NULL,
            trace_count INTEGER,
            event_count INTEGER,
            unique_activities INTEGER,
            time_range_start TEXT,
            time_range_end TEXT,
            activities_json TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Splits table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS splits (
            split_id TEXT PRIMARY KEY,
            log_id TEXT NOT NULL,
            train_ratio REAL,
            train_traces INTEGER,
            test_traces INTEGER,
            excluded_traces INTEGER,
            cutoff_time TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (log_id) REFERENCES event_logs(log_id)
        )
    """)
    
    # Models table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS models (
            model_id TEXT PRIMARY KEY,
            log_id TEXT NOT NULL,
            split_id TEXT NOT NULL,
            status TEXT DEFAULT 'pending',
            training_time_seconds REAL,
            metrics_json TEXT,
            model_path TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (log_id) REFERENCES event_logs(log_id),
            FOREIGN KEY (split_id) REFERENCES splits(split_id)
        )
    """)
    
    # Evaluations table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS evaluations (
            evaluation_id TEXT PRIMARY KEY,
            model_id TEXT NOT NULL,
            results_json TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (model_id) REFERENCES models(model_id)
        )
    """)
    
    # Simulator sessions table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS simulator_sessions (
            session_id TEXT PRIMARY KEY,
            model_id TEXT NOT NULL,
            trace_id TEXT,
            state_json TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (model_id) REFERENCES models(model_id)
        )
    """)
    
    conn.commit()
    conn.close()


def reset_demo():
    """
    Reset the database and clean up all data files for demo mode.
    This ensures a fresh start each time the server is started.
    """
    import shutil
    
    # Data directories to clean
    data_dir = Path(__file__).parent.parent / "data"
    uploads_dir = data_dir / "uploads"
    models_dir = data_dir / "models"
    splits_dir = data_dir / "splits"
    
    # Clear uploaded files
    if uploads_dir.exists():
        for f in uploads_dir.iterdir():
            if f.is_file():
                f.unlink()
    
    # Clear models
    if models_dir.exists():
        for f in models_dir.iterdir():
            if f.is_file():
                f.unlink()
            elif f.is_dir():
                shutil.rmtree(f)
    
    # Clear splits
    if splits_dir.exists():
        for f in splits_dir.iterdir():
            if f.is_file():
                f.unlink()
    
    # Clear database tables
    conn = get_connection()
    cursor = conn.cursor()
    
    # Delete all rows from tables (order matters for foreign keys)
    cursor.execute("DELETE FROM simulator_sessions")
    cursor.execute("DELETE FROM evaluations")
    cursor.execute("DELETE FROM models")
    cursor.execute("DELETE FROM splits")
    cursor.execute("DELETE FROM event_logs")
    
    conn.commit()
    conn.close()
    
    print("✓ Demo mode: cleared all data for fresh start")


# ─────────────────────────────────────────────────────────────────────────────
# EVENT LOG OPERATIONS
# ─────────────────────────────────────────────────────────────────────────────

def save_event_log(
    log_id: str,
    filename: str,
    trace_count: int,
    event_count: int,
    unique_activities: int,
    time_range_start: datetime,
    time_range_end: datetime,
    activities: List[str]
):
    """Save event log metadata to database."""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        INSERT OR REPLACE INTO event_logs 
        (log_id, filename, trace_count, event_count, unique_activities, 
         time_range_start, time_range_end, activities_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        log_id, filename, trace_count, event_count, unique_activities,
        time_range_start.isoformat(), time_range_end.isoformat(),
        json.dumps(activities)
    ))
    
    conn.commit()
    conn.close()


def get_event_log(log_id: str) -> Optional[Dict[str, Any]]:
    """Get event log metadata."""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT * FROM event_logs WHERE log_id = ?", (log_id,))
    row = cursor.fetchone()
    conn.close()
    
    if row:
        return dict(row)
    return None


# ─────────────────────────────────────────────────────────────────────────────
# SPLIT OPERATIONS
# ─────────────────────────────────────────────────────────────────────────────

def save_split(
    split_id: str,
    log_id: str,
    train_ratio: float,
    train_traces: int,
    test_traces: int,
    excluded_traces: int,
    cutoff_time: datetime
):
    """Save split metadata to database."""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        INSERT OR REPLACE INTO splits 
        (split_id, log_id, train_ratio, train_traces, test_traces, 
         excluded_traces, cutoff_time)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        split_id, log_id, train_ratio, train_traces, test_traces,
        excluded_traces, cutoff_time.isoformat()
    ))
    
    conn.commit()
    conn.close()


def get_split(split_id: str) -> Optional[Dict[str, Any]]:
    """Get split metadata."""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT * FROM splits WHERE split_id = ?", (split_id,))
    row = cursor.fetchone()
    conn.close()
    
    if row:
        return dict(row)
    return None


def get_latest_split_for_log(log_id: str) -> Optional[Dict[str, Any]]:
    """Get the most recent split for a log."""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT * FROM splits 
        WHERE log_id = ? 
        ORDER BY created_at DESC 
        LIMIT 1
    """, (log_id,))
    row = cursor.fetchone()
    conn.close()
    
    if row:
        return dict(row)
    return None


# ─────────────────────────────────────────────────────────────────────────────
# MODEL OPERATIONS
# ─────────────────────────────────────────────────────────────────────────────

def save_model(
    model_id: str,
    log_id: str,
    split_id: str,
    status: str = "pending",
    training_time_seconds: float = 0,
    metrics: Optional[Dict] = None,
    model_path: Optional[str] = None
):
    """Save model metadata to database."""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        INSERT OR REPLACE INTO models 
        (model_id, log_id, split_id, status, training_time_seconds, 
         metrics_json, model_path)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        model_id, log_id, split_id, status, training_time_seconds,
        json.dumps(metrics) if metrics else None, model_path
    ))
    
    conn.commit()
    conn.close()


def update_model_status(model_id: str, status: str, **kwargs):
    """Update model training status."""
    conn = get_connection()
    cursor = conn.cursor()
    
    updates = ["status = ?"]
    values = [status]
    
    if "training_time_seconds" in kwargs:
        updates.append("training_time_seconds = ?")
        values.append(kwargs["training_time_seconds"])
    
    if "metrics" in kwargs:
        updates.append("metrics_json = ?")
        values.append(json.dumps(kwargs["metrics"]))
    
    if "model_path" in kwargs:
        updates.append("model_path = ?")
        values.append(kwargs["model_path"])
    
    values.append(model_id)
    
    cursor.execute(f"""
        UPDATE models 
        SET {', '.join(updates)}
        WHERE model_id = ?
    """, values)
    
    conn.commit()
    conn.close()


def get_model(model_id: str) -> Optional[Dict[str, Any]]:
    """Get model metadata."""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT * FROM models WHERE model_id = ?", (model_id,))
    row = cursor.fetchone()
    conn.close()
    
    if row:
        result = dict(row)
        if result.get("metrics_json"):
            result["metrics"] = json.loads(result["metrics_json"])
        return result
    return None


# ─────────────────────────────────────────────────────────────────────────────
# EVALUATION OPERATIONS
# ─────────────────────────────────────────────────────────────────────────────

def save_evaluation(evaluation_id: str, model_id: str, results: Dict):
    """Save evaluation results."""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        INSERT OR REPLACE INTO evaluations 
        (evaluation_id, model_id, results_json)
        VALUES (?, ?, ?)
    """, (evaluation_id, model_id, json.dumps(results)))
    
    conn.commit()
    conn.close()


def get_evaluation(model_id: str) -> Optional[Dict[str, Any]]:
    """Get evaluation results for a model."""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT * FROM evaluations 
        WHERE model_id = ? 
        ORDER BY created_at DESC 
        LIMIT 1
    """, (model_id,))
    row = cursor.fetchone()
    conn.close()
    
    if row:
        result = dict(row)
        if result.get("results_json"):
            result["results"] = json.loads(result["results_json"])
        return result
    return None


# ─────────────────────────────────────────────────────────────────────────────
# SIMULATOR SESSION OPERATIONS
# ─────────────────────────────────────────────────────────────────────────────

def save_simulator_session(session_id: str, model_id: str, trace_id: str, state: Dict):
    """Save simulator session state."""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        INSERT OR REPLACE INTO simulator_sessions 
        (session_id, model_id, trace_id, state_json, updated_at)
        VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
    """, (session_id, model_id, trace_id, json.dumps(state)))
    
    conn.commit()
    conn.close()


def get_simulator_session(session_id: str) -> Optional[Dict[str, Any]]:
    """Get simulator session state."""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT * FROM simulator_sessions WHERE session_id = ?", (session_id,))
    row = cursor.fetchone()
    conn.close()
    
    if row:
        result = dict(row)
        if result.get("state_json"):
            result["state"] = json.loads(result["state_json"])
        return result
    return None


# Initialize database on module import
init_db()
