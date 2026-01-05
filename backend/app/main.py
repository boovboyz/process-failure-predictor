"""
FastAPI main application.
"""
import os
from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Load environment variables from .env file
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)

from app.api import upload, split, train, evaluate, predict, simulator
from app.database import init_db, reset_demo

# Initialize FastAPI app
app = FastAPI(
    title="Process Failure Predictor",
    description="Domain-agnostic process failure prediction using XES event logs",
    version="1.0.0"
)

# CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(upload.router, prefix="/api", tags=["Upload"])
app.include_router(split.router, prefix="/api", tags=["Split"])
app.include_router(train.router, prefix="/api", tags=["Training"])
app.include_router(evaluate.router, prefix="/api", tags=["Evaluation"])
app.include_router(predict.router, prefix="/api", tags=["Prediction"])
app.include_router(simulator.router, prefix="/api", tags=["Simulator"])


@app.on_event("startup")
async def startup():
    """Initialize database and reset demo data on startup."""
    init_db()
    reset_demo()
    
    # Initialize recommendation engine
    from app.core.recommendations import get_recommendation_engine
    engine = get_recommendation_engine()
    if engine.enabled:
        print("✓ LLM recommendations enabled (Claude API)")
    else:
        print("⚠ LLM recommendations disabled (using fallback rules)")


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Process Failure Predictor API",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}
