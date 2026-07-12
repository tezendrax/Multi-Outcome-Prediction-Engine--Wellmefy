from fastapi import FastAPI, HTTPException, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from typing import Dict, Any

from mope.config import settings
from mope.db import init_db, get_db, ModelRegistry
from mope.schemas import PredictionsResponse, PredictionTriggerRequest
from mope.inference import engine_singleton

from fastapi.staticfiles import StaticFiles
import os

app = FastAPI(
    title=settings.PROJECT_NAME,
    description="Multi-Outcome Prediction Engine (MOPE) evaluating student wellness risks.",
    version="1.0.0"
)

# Configure CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def on_startup():
    """Initializes local databases and registries on startup."""
    init_db()
    # Try to pre-load active models if they exist in the registry
    db = SessionLocal = next(get_db())
    try:
        engine_singleton.load_active_models(db)
    except Exception as e:
        # It's fine to fail startup loading if training hasn't run yet
        print(f"Warning on startup loading: {e}. Inference will be unavailable until models are trained.")
    finally:
        db.close()

@app.get("/health")
def health_check(db: Session = Depends(get_db)):
    """Health check endpoint showing current model version registry details."""
    active_models = db.query(ModelRegistry).filter(ModelRegistry.active == True).all()
    models_status = {}
    for m in active_models:
        models_status[m.model_type] = {
            "version": m.version,
            "created_at": m.created_at.isoformat()
        }
        
    return {
        "status": "healthy",
        "service": settings.PROJECT_NAME,
        "active_version": engine_singleton.active_version,
        "registered_models": models_status
    }

@app.get("/api/v1/predictions/mope", response_model=PredictionsResponse)
def get_mope_predictions(
    student_id: str = Query(..., description="Unique ID of the student"),
    semester_week: int = Query(6, ge=1, le=18, description="Current week of the semester"),
    db: Session = Depends(get_db)
):
    """
    Fetches multi-outcome predictions for a student.
    Uses cached predictions if available. Otherwise, performs on-demand inference.
    """
    try:
        # Calculate days_to_midterms based on week: midterms week is week 8
        days_to_midterms = max(0.0, (8.0 - semester_week) * 7.0)
        
        result = engine_singleton.run_prediction(
            student_id=student_id,
            days_to_midterms=days_to_midterms,
            force_recompute=False
        )
        return result
    except RuntimeError as re:
        raise HTTPException(status_code=503, detail=str(re))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/api/v1/predictions/mope/trigger", response_model=PredictionsResponse)
def trigger_mope_predictions(
    request: PredictionTriggerRequest,
    db: Session = Depends(get_db)
):
    """
    Explicitly triggers prediction recalculation.
    Ignores/overwrites cache, saves results to the local SQLite DB, updates the cache, and returns.
    """
    try:
        # Resolve days to midterms
        days_to_midterms = request.temporal_metadata.days_to_midterms
        if days_to_midterms is None:
            semester_week = request.temporal_metadata.semester_week
            days_to_midterms = max(0.0, (8.0 - semester_week) * 7.0)
            
        result = engine_singleton.run_prediction(
            student_id=request.student_id,
            days_to_midterms=days_to_midterms,
            force_recompute=True
        )
        return result
    except RuntimeError as re:
        raise HTTPException(status_code=503, detail=str(re))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction trigger failed: {str(e)}")

# Mount static files for the dashboard frontend
frontend_path = os.path.normpath(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend"))
os.makedirs(frontend_path, exist_ok=True)
app.mount("/", StaticFiles(directory=frontend_path, html=True), name="static")
