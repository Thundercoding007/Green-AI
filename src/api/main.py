# File: src/api/main.py
# FastAPI Backend for GreenAI Email Classifier (rewritten)
# - Adds router-info endpoint (calibrated temps + per-class thresholds)
# - Prefer JSON config at data/models/cascade/green_ai_config.json
# - Backwards-compatible with existing cascade classifier behavior

from fastapi import FastAPI, HTTPException, Depends, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
import time
from pathlib import Path
import json
import joblib

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

# CORS middleware (allow all for local/dev)
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
    emails: List[EmailRequest] = Field(..., max_items=100)


class StatsResponse(BaseModel):
    total_inferences: int
    cascade_accuracy: Optional[float]
    model_distribution: Dict[str, float]
    energy_saved_percent: Optional[float]
    co2_saved_grams: Optional[float]
    avg_inference_time_ms: float


class HealthResponse(BaseModel):
    status: str
    models_loaded: bool
    cascade_ready: bool
    database_connected: bool
    timestamp: datetime


# NOTE: This ThresholdsRequest is preserved for backward compatibility,
# but manual slider updates will be removed on the dashboard.
class ThresholdsRequest(BaseModel):
    green_threshold: float = Field(..., ge=0.5, le=1.0)
    medium_threshold: float = Field(..., ge=0.5, le=1.0)


# Database dependency
def get_db():
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
    global cascade_classifier, energy_tracker
    print("🚀 Starting GreenAI API...")
    print(f"📁 Project Root: {Config.PROJECT_ROOT}")
    print(f"🗃️ Database Path: {Config.DATABASE_PATH}")

    try:
        print("📦 Loading models into CascadeClassifier...")
        cascade_classifier = CascadeClassifier.load_models_and_create(
            Config.GREEN_MODEL_PATH,
            Config.MEDIUM_MODEL_PATH,
            Config.HEAVY_MODEL_PATH,
            config_path=Config.MODELS_DIR / "cascade",
        )

        # Create energy tracker
        energy_tracker = CascadeEnergyTracker(output_dir=Config.EMISSIONS_DIR)

        print("✅ Models loaded successfully!")
        # Try to print a helpful summary of thresholds if available
        # cascade_classifier may have scalar thresholds older style
        try:
            print(f"   green_threshold: {getattr(cascade_classifier, 'green_threshold', 'N/A')}")
            print(f"   medium_threshold: {getattr(cascade_classifier, 'medium_threshold', 'N/A')}")
        except Exception:
            pass

    except Exception as e:
        print(f"❌ Error loading models: {e}")
        cascade_classifier = None
        energy_tracker = None
        print("   API started but classification endpoints will return 503 until models are available.")


# Routes
@app.get("/", tags=["Root"])
async def root():
    return {
        "message": "🌿 Welcome to GreenAI Email Classifier API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
    }


from sqlalchemy import text  # local import kept for DB health probe


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
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
    if cascade_classifier is None:
        raise HTTPException(status_code=503, detail="Models not loaded")

    try:
        start_time = time.time()
        result = cascade_classifier.predict_single(request.text, return_details=True)

        energy_kwh = None
        co2_grams = None
        if request.track_energy and energy_tracker:
            energy_log = energy_tracker.log_cascade_inference(result, actual_label=None)
            energy_kwh = energy_log.get("energy_kwh")
            co2_grams = energy_log.get("co2_grams")

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
                print(f"⚠️ DB write failed: {e}")

        # Prepare probabilities dict (make serializable)
        probs = result.get("probabilities", {})
        # If probs is numpy array-like, try to convert to dict[label->value] when labels are known
        try:
            if isinstance(probs, (list, tuple)) or (hasattr(probs, "tolist") and not isinstance(probs, dict)):
                # try mapping using classes from green/medium/heavy if possible
                # fallback: enumerate
                prob_list = list(probs.tolist()) if hasattr(probs, "tolist") else list(probs)
                probs = {str(i): float(v) for i, v in enumerate(prob_list)}
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
    if cascade_classifier is None:
        raise HTTPException(status_code=503, detail="Models not loaded")

    if len(request.emails) > 100:
        raise HTTPException(status_code=400, detail="Maximum 100 emails per batch")

    results: List[Dict[str, Any]] = []
    db = None
    if DB_AVAILABLE:
        db = SessionLocal()

    try:
        for email_req in request.emails:
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
    if not DB_AVAILABLE:
        raise HTTPException(status_code=503, detail="Database not available")
    db = SessionLocal()
    try:
        stats = get_model_statistics(db, model_name)
        return stats or {"message": "No data available"}
    finally:
        db.close()


# -------------------------------------------------------------------------
# New endpoint: return router (calibration + per-class thresholds) info
# Priority:
# 1) data/models/cascade/green_ai_config.json
# 2) data/models/cascade/cascade_config.pkl (joblib) if present and contains thresholds
# 3) fall back to cascade_classifier scalar thresholds (older behavior)
# -------------------------------------------------------------------------
@app.get("/config/router-info", tags=["Configuration"])
async def get_router_info():
    # 1) Preferred: JSON config file
    cfg_path = Config.MODELS_DIR / "cascade" / "green_ai_config.json"
    try:
        if cfg_path.exists():
            with open(cfg_path, "r") as f:
                cfg = json.load(f)
            # ensure minimal fields
            cfg.setdefault("classes", Config.CLASSES)
            cfg.setdefault("num_classes", len(cfg.get("classes", Config.CLASSES)))
            return cfg
    except Exception as e:
        print(f"⚠️ Could not load green_ai_config.json: {e}")

    # 2) Fallback: try cascade_config.pkl saved by CascadeClassifier.save
    pkl_path = Config.MODELS_DIR / "cascade" / "cascade_config.pkl"
    try:
        if pkl_path.exists():
            cfg_loaded = joblib.load(pkl_path)
            # Normalize keys
            out = {
                "temperature_lr": cfg_loaded.get("temperature_lr"),
                "temperature_med": cfg_loaded.get("temperature_med"),
                "thresholds_lr": cfg_loaded.get("thresholds_lr", {}),
                "thresholds_med": cfg_loaded.get("thresholds_med", {}),
                "num_classes": cfg_loaded.get("num_classes", len(Config.CLASSES)),
                "classes": cfg_loaded.get("classes", Config.CLASSES),
                "validation_stats": cfg_loaded.get("validation_stats"),
            }
            # If thresholds are scalars, expand per-class
            if isinstance(out["thresholds_lr"], (float, int)):
                out["thresholds_lr"] = {str(i): float(out["thresholds_lr"]) for i in range(out["num_classes"])}
            if isinstance(out["thresholds_med"], (float, int)):
                out["thresholds_med"] = {str(i): float(out["thresholds_med"]) for i in range(out["num_classes"])}
            return out
    except Exception as e:
        print(f"⚠️ Could not load cascade_config.pkl: {e}")

    # 3) As last resort, try to read simple fields from loaded cascade_classifier
    if cascade_classifier is not None:
        try:
            t_green = getattr(cascade_classifier, "green_threshold", Config.GREEN_THRESHOLD)
            t_med = getattr(cascade_classifier, "medium_threshold", Config.MEDIUM_THRESHOLD)
            out = {
                "temperature_lr": None,
                "temperature_med": None,
                "thresholds_lr": {str(i): float(t_green) for i in range(len(Config.CLASSES))},
                "thresholds_med": {str(i): float(t_med) for i in range(len(Config.CLASSES))},
                "num_classes": len(Config.CLASSES),
                "classes": Config.CLASSES,
                "validation_stats": None,
            }
            return out
        except Exception as e:
            print(f"⚠️ Could not extract thresholds from cascade_classifier: {e}")

    raise HTTPException(status_code=503, detail="Router configuration not available. Run fit step and save green_ai_config.json to data/models/cascade/.")


@app.get("/config", tags=["Configuration"])
async def get_configuration():
    """Return a compact configuration summary (keeps backward-compatible fields)."""
    if cascade_classifier is None:
        raise HTTPException(status_code=503, detail="Models not loaded")

    # load router-info if present
    router_info = None
    try:
        router_info = (Config.MODELS_DIR / "cascade" / "green_ai_config.json")
        if router_info.exists():
            with open(router_info, "r") as f:
                router_info = json.load(f)
        else:
            router_info = None
    except Exception:
        router_info = None

    return {
        "green_threshold": float(getattr(cascade_classifier, "green_threshold", Config.GREEN_THRESHOLD)),
        "medium_threshold": float(getattr(cascade_classifier, "medium_threshold", Config.MEDIUM_THRESHOLD)),
        "classes": Config.CLASSES,
        "model_info": {
            "green": {
                "type": "TF-IDF + Logistic Regression",
                "size_mb": cascade_classifier.green_model.estimate_size() if cascade_classifier else 0,
            },
            "medium": {
                "type": "DistilBERT",
                "size_mb": cascade_classifier.medium_model.estimate_size() if cascade_classifier else 0,
                "max_length": getattr(cascade_classifier.medium_model, "max_length", None),
            },
            "heavy": {
                "type": "DeBERTa-v3",
                "size_mb": cascade_classifier.heavy_model.estimate_size() if cascade_classifier else 0,
                "max_length": getattr(cascade_classifier.heavy_model, "max_length", None),
            },
        },
        "router_present": bool(router_info),
    }


@app.post("/config/thresholds", tags=["Configuration"])
async def update_thresholds(payload: ThresholdsRequest = Body(...)):
    """
    Backwards-compatible endpoint to update scalar thresholds (kept for legacy UX).
    Note: prefer updating per-class thresholds via offline fit and saving green_ai_config.json.
    """
    if cascade_classifier is None:
        raise HTTPException(status_code=503, detail="Models not loaded")

    if payload.medium_threshold > payload.green_threshold:
        raise HTTPException(status_code=400, detail="Medium threshold cannot be higher than green threshold")

    old_green = float(getattr(cascade_classifier, "green_threshold", Config.GREEN_THRESHOLD))
    old_medium = float(getattr(cascade_classifier, "medium_threshold", Config.MEDIUM_THRESHOLD))

    cascade_classifier.green_threshold = float(payload.green_threshold)
    cascade_classifier.medium_threshold = float(payload.medium_threshold)

    try:
        # Save old-style cascade_config.pkl for backward compatibility
        cascade_classifier.save(Config.MODELS_DIR / "cascade")
    except Exception as e:
        print(f"⚠️ Could not save cascade config: {e}")

    return {
        "message": "Thresholds updated (scalar) successfully",
        "old_thresholds": {"green": old_green, "medium": old_medium},
        "new_thresholds": {"green": payload.green_threshold, "medium": payload.medium_threshold},
    }


@app.get("/models/info", tags=["Models"])
async def get_models_info():
    if cascade_classifier is None:
        raise HTTPException(status_code=503, detail="Models not loaded")

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
