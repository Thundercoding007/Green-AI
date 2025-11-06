# File: src/api/main.py
# FastAPI Backend for GreenAI Email Classifier

from fastapi import FastAPI, HTTPException, Depends, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
import time
from pathlib import Path

from src.config import Config
from src.models.cascade import CascadeClassifier
from src.models.green_model import GreenModel
from src.models.medium_model import MediumModel
from src.models.heavy_model import HeavyModel
from src.utils.energy_tracker import CascadeEnergyTracker

# Optional database imports
try:
    from src.database import (
        SessionLocal,
        insert_inference_log,
        get_model_statistics,
        calculate_energy_savings,
    )
    DB_AVAILABLE = True
except Exception:
    DB_AVAILABLE = False

# Initialize FastAPI app
app = FastAPI(
    title="🌿 GreenAI Email Classifier API",
    description="Green AI-powered email classification with energy tracking",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for models
cascade_classifier: Optional[CascadeClassifier] = None
energy_tracker: Optional[CascadeEnergyTracker] = None

# -----------------------------
# Pydantic models
# -----------------------------
class EmailRequest(BaseModel):
    """Request model for email classification"""
    text: str = Field(..., description="Email text to classify", min_length=10)
    email_id: Optional[str] = Field(None, description="Optional email ID for tracking")
    track_energy: bool = Field(True, description="Whether to track energy consumption")

    class Config:
        schema_extra = {
            "example": {
                "text": "Meeting scheduled for tomorrow at 10am. Please review the agenda.",
                "email_id": "email_123",
                "track_energy": True,
            }
        }


class ClassificationResponse(BaseModel):
    """Response model for classification"""
    email_id: Optional[str]
    prediction: str
    confidence: float
    probabilities: Dict[str, float]
    model_used: str
    cascade_path: str
    inference_time_ms: float
    energy_kwh: Optional[float] = None
    co2_grams: Optional[float] = None
    timestamp: datetime


class BatchEmailRequest(BaseModel):
    """Request model for batch classification"""
    emails: List[EmailRequest] = Field(..., max_items=100)

    class Config:
        schema_extra = {
            "example": {
                "emails": [
                    {"text": "Meeting tomorrow at 10am", "email_id": "1"},
                    {"text": "SALE! 50% off everything", "email_id": "2"},
                ]
            }
        }


class StatsResponse(BaseModel):
    """Response model for statistics"""
    total_inferences: int
    cascade_accuracy: Optional[float]
    model_distribution: Dict[str, float]
    energy_saved_percent: Optional[float]
    co2_saved_grams: Optional[float]
    avg_inference_time_ms: float


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    models_loaded: bool
    cascade_ready: bool
    database_connected: bool
    timestamp: datetime


class ThresholdsRequest(BaseModel):
    green_threshold: float = Field(..., ge=0.5, le=1.0)
    medium_threshold: float = Field(..., ge=0.5, le=1.0)


# Database dependency
def get_db():
    """Get database session (if DB available)."""
    if not DB_AVAILABLE:
        raise HTTPException(status_code=503, detail="Database not available")
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# Startup event
@app.on_event("startup")
async def startup_event():
    """Load models on startup"""
    global cascade_classifier, energy_tracker

    print("🚀 Starting GreenAI API...")
    print(f"📁 Project Root: {Config.PROJECT_ROOT}")
    print(f"🗃️ Database Path: {Config.DATABASE_PATH}")


    try:
        # Load cascade classifier
        print("📦 Loading models...")
        cascade_classifier = CascadeClassifier.load_models_and_create(
            Config.GREEN_MODEL_PATH,
            Config.MEDIUM_MODEL_PATH,
            Config.HEAVY_MODEL_PATH,
            config_path=Config.MODELS_DIR / "cascade",
        )

        # Initialize energy tracker
        energy_tracker = CascadeEnergyTracker(output_dir=Config.EMISSIONS_DIR)

        print("✅ Models loaded successfully!")
        print(f"   Green threshold: {cascade_classifier.green_threshold:.2f}")
        print(f"   Medium threshold: {cascade_classifier.medium_threshold:.2f}")

    except Exception as e:
        print(f"❌ Error loading models: {e}")
        cascade_classifier = None
        energy_tracker = None
        print("   API started but classification endpoints will return 503 until models are available.")


# Routes
@app.get("/", tags=["Root"])
async def root():
    """Root endpoint"""
    return {
        "message": "🌿 Welcome to GreenAI Email Classifier API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
    }


from sqlalchemy import text  # <-- add this import at the top with other imports

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint"""
    db_connected = True
    try:
        db = SessionLocal()
        db.execute(text("SELECT name FROM sqlite_master WHERE type='table' LIMIT 1;"))
        db.close()
    except Exception as e:
        print(f"⚠️ DB connection failed: {e}")
        db_connected = False

    return HealthResponse(
        status="healthy" if cascade_classifier and db_connected else "degraded",
        models_loaded=cascade_classifier is not None,
        cascade_ready=cascade_classifier is not None,
        database_connected=db_connected,
        timestamp=datetime.utcnow()
    )




@app.post("/classify", response_model=ClassificationResponse, tags=["Classification"])
async def classify_email(request: EmailRequest, db=Depends(get_db) if DB_AVAILABLE else None):
    """
    Classify a single email using cascade classifier
    """
    if cascade_classifier is None:
        raise HTTPException(status_code=503, detail="Models not loaded")

    try:
        # Get prediction
        start_time = time.time()
        result = cascade_classifier.predict_single(request.text, return_details=True)

        energy_kwh = None
        co2_grams = None

        # Track energy if requested
        if request.track_energy and energy_tracker:
            energy_log = energy_tracker.log_cascade_inference(result, actual_label=None)
            energy_kwh = energy_log.get("energy_kwh")
            co2_grams = energy_log.get("co2_grams")

        # Log to DB if available
        if DB_AVAILABLE and db is not None:
            try:
                log_data = {
                    "email_id": request.email_id or f"api_{int(time.time() * 1000)}",
                    "email_text_preview": request.text[:200],
                    "predicted_class": result["prediction"],
                    "confidence": result["confidence"],
                    "model_used": result["model_used"],
                    "energy_kwh": energy_kwh or 0.0,
                    "co2_grams": co2_grams or 0.0,
                    "inference_time_ms": result.get("total_time_ms", 0),
                    "cascade_path": result.get("cascade_path", ""),
                }
                insert_inference_log(db, log_data)
            except Exception as e:
                # Do not fail the request if DB write fails; just log server-side
                print(f"⚠️ DB write failed: {e}")

        # Prepare probabilities dict (ensure serializable)
        probs = result.get("probabilities", {})
        if hasattr(probs, "tolist"):
            try:
                probs = {k: float(v) for k, v in enumerate(probs.tolist())}
            except Exception:
                pass

        response = ClassificationResponse(
            email_id=request.email_id,
            prediction=result["prediction"],
            confidence=float(result["confidence"]),
            probabilities={k: float(v) for k, v in probs.items()},
            model_used=result["model_used"],
            cascade_path=result.get("cascade_path", ""),
            inference_time_ms=float(result.get("total_time_ms", 0)),
            energy_kwh=energy_kwh,
            co2_grams=co2_grams,
            timestamp=datetime.utcnow(),
        )

        return response

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Classification error: {str(e)}")


@app.post("/classify/batch", tags=["Classification"])
async def classify_batch(request: BatchEmailRequest):
    """
    Classify multiple emails in batch
    """
    if cascade_classifier is None:
        raise HTTPException(status_code=503, detail="Models not loaded")

    if len(request.emails) > 100:
        raise HTTPException(status_code=400, detail="Maximum 100 emails per batch")

    results: List[Dict[str, Any]] = []
    # Use a DB session only if DB is available
    db = None
    if DB_AVAILABLE:
        db = SessionLocal()

    try:
        for email_req in request.emails:
            # Use the same logic as /classify but inline to avoid double-dependency issues
            try:
                result = cascade_classifier.predict_single(email_req.text, return_details=True)

                energy_kwh = None
                co2_grams = None
                if email_req.track_energy and energy_tracker:
                    energy_log = energy_tracker.log_cascade_inference(result, actual_label=None)
                    energy_kwh = energy_log.get("energy_kwh")
                    co2_grams = energy_log.get("co2_grams")

                if DB_AVAILABLE and db is not None:
                    try:
                        log_data = {
                            "email_id": email_req.email_id or f"batch_{int(time.time() * 1000)}",
                            "email_text_preview": email_req.text[:200],
                            "predicted_class": result["prediction"],
                            "confidence": result["confidence"],
                            "model_used": result["model_used"],
                            "energy_kwh": energy_kwh or 0.0,
                            "co2_grams": co2_grams or 0.0,
                            "inference_time_ms": result.get("total_time_ms", 0),
                            "cascade_path": result.get("cascade_path", ""),
                        }
                        insert_inference_log(db, log_data)
                    except Exception as e:
                        print(f"⚠️ DB write failed (batch): {e}")

                results.append({
                    "email_id": email_req.email_id,
                    "prediction": result["prediction"],
                    "confidence": float(result["confidence"]),
                    "model_used": result["model_used"],
                    "cascade_path": result.get("cascade_path"),
                    "inference_time_ms": float(result.get("total_time_ms", 0)),
                    "energy_kwh": energy_kwh,
                    "co2_grams": co2_grams,
                })

            except Exception as e:
                results.append({"email_id": email_req.email_id, "error": str(e)})

        successful = len([r for r in results if "error" not in r])
        return {"total": len(request.emails), "successful": successful, "results": results}
    finally:
        if DB_AVAILABLE and db is not None:
            db.close()


@app.get("/stats", response_model=StatsResponse, tags=["Statistics"])
async def get_statistics():
    """
    Get overall system statistics
    """
    try:
        cascade_stats = cascade_classifier.get_statistics() if cascade_classifier else {}
        db_stats = {}
        energy_savings = {}
        if DB_AVAILABLE:
            db = SessionLocal()
            try:
                db_stats = get_model_statistics(db) or {}
                energy_savings = calculate_energy_savings(db) or {}
            finally:
                db.close()

        total = int(cascade_stats.get("total_inferences", 0))
        return StatsResponse(
            total_inferences=total,
            cascade_accuracy=None,
            model_distribution={
                "green": float(cascade_stats.get("green_usage_pct", 0)),
                "medium": float(cascade_stats.get("medium_usage_pct", 0)),
                "heavy": float(cascade_stats.get("heavy_usage_pct", 0)),
            },
            energy_saved_percent=energy_savings.get("energy_saved_percent") if energy_savings else None,
            co2_saved_grams=energy_savings.get("cascade_co2_grams") if energy_savings else None,
            avg_inference_time_ms=float(db_stats.get("avg_inference_time_ms", 0)) if db_stats else 0,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Statistics error: {str(e)}")


@app.get("/stats/models", tags=["Statistics"])
async def get_model_stats(model_name: Optional[str] = None):
    """
    Get statistics for specific model or all models
    """
    if not DB_AVAILABLE:
        raise HTTPException(status_code=503, detail="Database not available")
    db = SessionLocal()
    try:
        stats = get_model_statistics(db, model_name)
        return stats or {"message": "No data available"}
    finally:
        db.close()


@app.get("/config", tags=["Configuration"])
async def get_configuration():
    """Get current cascade configuration"""
    if cascade_classifier is None:
        raise HTTPException(status_code=503, detail="Models not loaded")

    return {
        "green_threshold": float(cascade_classifier.green_threshold),
        "medium_threshold": float(cascade_classifier.medium_threshold),
        "classes": Config.CLASSES,
        "model_info": {
            "green": {
                "type": "TF-IDF + Logistic Regression",
                "size_mb": cascade_classifier.green_model.estimate_size(),
            },
            "medium": {
                "type": "DistilBERT",
                "size_mb": cascade_classifier.medium_model.estimate_size(),
                "max_length": getattr(cascade_classifier.medium_model, "max_length", None),
            },
            "heavy": {
                "type": "DeBERTa-v3",
                "size_mb": cascade_classifier.heavy_model.estimate_size(),
                "max_length": getattr(cascade_classifier.heavy_model, "max_length", None),
            },
        },
    }


@app.post("/config/thresholds", tags=["Configuration"])
async def update_thresholds(payload: ThresholdsRequest = Body(...)):
    """
    Update cascade thresholds (payload body):
    {
      "green_threshold": 0.85,
      "medium_threshold": 0.8
    }
    """
    if cascade_classifier is None:
        raise HTTPException(status_code=503, detail="Models not loaded")

    if payload.medium_threshold > payload.green_threshold:
        raise HTTPException(status_code=400, detail="Medium threshold cannot be higher than green threshold")

    old_green = float(cascade_classifier.green_threshold)
    old_medium = float(cascade_classifier.medium_threshold)

    cascade_classifier.green_threshold = float(payload.green_threshold)
    cascade_classifier.medium_threshold = float(payload.medium_threshold)

    # Save configuration
    try:
        cascade_classifier.save(Config.MODELS_DIR / "cascade")
    except Exception as e:
        print(f"⚠️ Could not save cascade config: {e}")

    return {
        "message": "Thresholds updated successfully",
        "old_thresholds": {"green": old_green, "medium": old_medium},
        "new_thresholds": {"green": payload.green_threshold, "medium": payload.medium_threshold},
    }


@app.get("/models/info", tags=["Models"])
async def get_models_info():
    """Get information about loaded models"""
    if cascade_classifier is None:
        raise HTTPException(status_code=503, detail="Models not loaded")

    # Convert numpy arrays to lists where necessary
    green_classes = getattr(cascade_classifier.green_model, "classes_", None)
    if hasattr(green_classes, "tolist"):
        green_classes = green_classes.tolist()

    return {
        "green_model": {
            "type": "TF-IDF + Logistic Regression",
            "size_mb": cascade_classifier.green_model.estimate_size(),
            "features": len(cascade_classifier.green_model.feature_names) if cascade_classifier.green_model.feature_names is not None else 0,
            "classes": green_classes,
        },
        "medium_model": {
            "type": "DistilBERT",
            "size_mb": cascade_classifier.medium_model.estimate_size(),
            "max_length": getattr(cascade_classifier.medium_model, "max_length", None),
            "classes": getattr(cascade_classifier.medium_model, "classes_", None),
        },
        "heavy_model": {
            "type": "DeBERTa-v3-base",
            "size_mb": cascade_classifier.heavy_model.estimate_size(),
            "max_length": getattr(cascade_classifier.heavy_model, "max_length", None),
            "classes": getattr(cascade_classifier.heavy_model, "classes_", None),
        },
    }


# Global exception handler (returns JSONResponse)
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc), "timestamp": datetime.utcnow().isoformat()},
    )


# Run with: uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
if __name__ == "__main__":
    import uvicorn

    print("🚀 Starting GreenAI API Server...")
    print(f"📍 Host: {getattr(Config, 'API_HOST', '0.0.0.0')}")
    print(f"🔌 Port: {getattr(Config, 'API_PORT', 8000)}")
    print(f"📚 Docs: http://{getattr(Config, 'API_HOST', '127.0.0.1')}:{getattr(Config, 'API_PORT', 8000)}/docs")

    uvicorn.run(app, host=getattr(Config, "API_HOST", "0.0.0.0"), port=getattr(Config, "API_PORT", 8000), log_level="info")
