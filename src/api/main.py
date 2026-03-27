# File: src/api/main.py
# FastAPI Backend for GreenAI Email Classifier (GreenRouter version)
# - Uses GreenRouter (per-class thresholds + temperature scaling)
# - Backwards-compatible JSON responses and endpoints

from fastapi import FastAPI, HTTPException, Depends, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime
import time
from pathlib import Path
import json
import joblib
import traceback

from src.config import Config
from src.models.green_model import GreenModel
from src.models.medium_model import MediumModel
from src.models.heavy_model import HeavyModel
from src.models.green_router import GreenRouter
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
    description="Green AI-powered email classification with energy tracking (GreenRouter)",
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

# Global variables for models / router / energy tracker
router: Optional[GreenRouter] = None
green_model: Optional[GreenModel] = None
medium_model: Optional[MediumModel] = None
heavy_model: Optional[HeavyModel] = None
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

    # NEW: per-model confidences & per-model probability arrays (for dashboard transparency)
    per_model_confidences: Optional[Dict[str, float]] = None
    per_model_probs: Optional[Dict[str, Dict[str, float]]] = None



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


# -----------------------------
# Startup event: load models + router + energy tracker
# -----------------------------
@app.on_event("startup")
async def startup_event():
    global router, green_model, medium_model, heavy_model, energy_tracker
    print("🚀 Starting GreenAI API...")
    print(f"📁 Project Root: {Config.PROJECT_ROOT}")
    print(f"🗃️ Database Path: {Config.DATABASE_PATH}")

    try:
        print("📦 Loading Green / Medium / Heavy models...")
        green_model = GreenModel.load(Config.GREEN_MODEL_PATH)
        medium_model = MediumModel.load(Config.MEDIUM_MODEL_PATH)
        heavy_model = HeavyModel.load(Config.HEAVY_MODEL_PATH)

        # Try to load router config JSON (preferred)
        cfg_path = Config.MODELS_DIR / "cascade" / "green_ai_config.json"
        try:
            if cfg_path.exists():
                print(f"🔎 Loading router config from: {cfg_path}")
                router = GreenRouter.from_config_path(cfg_path, green_model, medium_model, heavy_model)
                print("✅ GreenRouter loaded from JSON config.")
            else:
                # fallback: try to load legacy cascade_config.pkl OR construct minimal config using scalar thresholds
                legacy_pkl = Config.MODELS_DIR / "cascade" / "cascade_config.pkl"
                if legacy_pkl.exists():
                    try:
                        print("🔎 Loading legacy cascade_config.pkl for thresholds...")
                        p = joblib.load(legacy_pkl)
                        # build a minimal green_ai_config-like dict
                        num_classes = p.get("num_classes", len(Config.CLASSES))
                        t_lr = p.get("temperature_lr", None)
                        t_med = p.get("temperature_med", None)
                        thresholds_lr = p.get("thresholds_lr", p.get("green_threshold", Config.GREEN_THRESHOLD))
                        thresholds_med = p.get("thresholds_med", p.get("medium_threshold", Config.MEDIUM_THRESHOLD))
                        # expand scalars if needed
                        if isinstance(thresholds_lr, (float, int)):
                            thresholds_lr = {str(i): float(thresholds_lr) for i in range(num_classes)}
                        if isinstance(thresholds_med, (float, int)):
                            thresholds_med = {str(i): float(thresholds_med) for i in range(num_classes)}
                        cfg = {
                            "temperature_lr": t_lr if t_lr is not None else 1.0,
                            "temperature_med": t_med if t_med is not None else 1.0,
                            "thresholds_lr": {str(i): float(thresholds_lr.get(str(i), thresholds_lr[i]) if isinstance(thresholds_lr, dict) else thresholds_lr) for i in range(num_classes)},
                            "thresholds_med": {str(i): float(thresholds_med.get(str(i), thresholds_med[i]) if isinstance(thresholds_med, dict) else thresholds_med) for i in range(num_classes)},
                            "num_classes": num_classes,
                            "classes": p.get("classes", Config.CLASSES),
                            "validation_stats": p.get("validation_stats", None),
                        }
                        router = GreenRouter(cfg, green_model, medium_model, heavy_model)
                        print("✅ GreenRouter built from legacy cascade_config.pkl.")
                    except Exception as e:
                        print(f"⚠️ Failed to load legacy cascade_config.pkl ({e}) — building minimal router.")
                        router = None
                if router is None:
                    # Build minimal config using scalar thresholds from Config
                    print("ℹ️ Building minimal router config from scalar thresholds (fallback).")
                    num_classes = len(Config.CLASSES)
                    cfg = {
                        "temperature_lr": 1.0,
                        "temperature_med": 1.0,
                        "thresholds_lr": {str(i): float(Config.GREEN_THRESHOLD) for i in range(num_classes)},
                        "thresholds_med": {str(i): float(Config.MEDIUM_THRESHOLD) for i in range(num_classes)},
                        "num_classes": num_classes,
                        "classes": Config.CLASSES,
                        "validation_stats": None,
                    }
                    router = GreenRouter(cfg, green_model, medium_model, heavy_model)
                    print("✅ GreenRouter created (fallback scalar thresholds).")
        except Exception as e:
            print(f"❌ Failed to initialize router from config: {e}")
            traceback.print_exc()
            router = None

        # Create energy tracker
        energy_tracker = CascadeEnergyTracker(output_dir=Config.EMISSIONS_DIR)

        print("✅ Models & energy tracker loaded.")
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        traceback.print_exc()
        router = None
        green_model = medium_model = heavy_model = None
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
        if DB_AVAILABLE:
            db = SessionLocal()
            # simple probe
            db.execute(text("SELECT name FROM sqlite_master WHERE type='table' LIMIT 1;"))
            db.close()
    except Exception as e:
        print(f"⚠️ DB connection failed: {e}")
        db_connected = False

    models_loaded = all([green_model is not None, medium_model is not None, heavy_model is not None])
    router_ready = router is not None

    return {
    "status": "healthy" if models_loaded and db_connected and router_ready else "degraded",
    "models_loaded": models_loaded,
    "cascade_ready": router_ready,  # kept for backward compat
    "router_ready": router_ready,   # NEW FIELD REQUIRED BY DASHBOARD
    "database_connected": db_connected,
    "timestamp": datetime.utcnow().isoformat(),
}



# -----------------------------
# Helper: map label index -> label string using available model metadata
# -----------------------------
def label_from_index(model_name: str, idx: int) -> str:
    """Map predicted integer index to label name using the appropriate model's metadata."""
    try:
        if model_name.lower() == "green" and green_model is not None:
            # green_model.classes_ is numpy array
            return str(green_model.classes_[int(idx)])
        if model_name.lower() in ("medium", "med"):
            if medium_model is not None:
                # medium_model.id2label may be dict
                if hasattr(medium_model, "id2label") and medium_model.id2label:
                    return str(medium_model.id2label[int(idx)])
                if hasattr(medium_model, "classes_") and medium_model.classes_ is not None:
                    return str(medium_model.classes_[int(idx)])
        if model_name.lower() == "heavy" and heavy_model is not None:
            if hasattr(heavy_model, "id2label") and heavy_model.id2label:
                return str(heavy_model.id2label[int(idx)])
            if hasattr(heavy_model, "classes_") and heavy_model.classes_ is not None:
                return str(heavy_model.classes_[int(idx)])
    except Exception:
        pass
    # fallback to global Config classes
    return str(Config.CLASSES[int(idx)]) if 0 <= int(idx) < len(Config.CLASSES) else str(idx)


# -----------------------------
# /classify
# -----------------------------
@app.post("/classify", response_model=ClassificationResponse, tags=["Classification"])
async def classify_email(request: EmailRequest, db=Depends(get_db) if DB_AVAILABLE else None):
    if router is None:
        raise HTTPException(status_code=503, detail="Models/router not loaded")

    try:
        start_time = time.time()
                # router.predict_single returns (pred_idx:int, model_used:str, details:dict)
        pred_idx, model_used, details = router.predict_single(request.text)
        inference_time_ms = (time.time() - start_time) * 1000

        # Map index -> label string
        prediction_label = label_from_index(model_used, int(pred_idx))

        # DETAILS structure (from patched router) contains:
        # details = {
        #   "probs": {"green":[...], "medium":[...], "heavy":[...]},
        #   "confidences": {"green":0.74, "medium":0.90, "heavy":0.88},
        #   "preds": {"green":0, "medium":0, "heavy":0}
        # }
        per_model_confidences = {}
        per_model_probs = {}
        chosen_probs = {}
        chosen_confidence = 0.0

        try:
            if details:
                # extract per-model confidences (if present)
                per_model_confidences = {k: float(v) for k, v in details.get("confidences", {}).items()}

                # convert each model's prob-array into a label->prob dict
                for model_key, prob_arr in details.get("probs", {}).items():
                    # pick labels from corresponding model metadata if available
                    labels = Config.CLASSES
                    if model_key == "green" and green_model is not None:
                        labels = list(map(str, green_model.classes_))
                    elif model_key in ("medium", "med") and medium_model is not None:
                        labels = [str(v) for v in getattr(medium_model, "id2label", getattr(medium_model, "classes_", Config.CLASSES))]
                    elif model_key == "heavy" and heavy_model is not None:
                        labels = [str(v) for v in getattr(heavy_model, "id2label", getattr(heavy_model, "classes_", Config.CLASSES))]

                    # if labels length mismatches, fallback to numeric keys
                    if len(labels) == len(prob_arr):
                        prob_map = {str(labels[i]): float(prob_arr[i]) for i in range(len(prob_arr))}
                    else:
                        prob_map = {str(i): float(prob_arr[i]) for i in range(len(prob_arr))}

                    per_model_probs[model_key] = prob_map

                # chosen model's probs and confidence
                chosen_probs = per_model_probs.get(model_used, {})
                chosen_confidence = float(per_model_confidences.get(model_used, max(chosen_probs.values()) if chosen_probs else 0.0))
        except Exception:
            # Fallback: try details["confidence"] or set zeros
            chosen_confidence = float(details.get("confidence", 0.0)) if details else 0.0

        # Build cascade-like result for backwards compat (energy tracker & DB)
        cascade_like_result = {
            "prediction": prediction_label,
            "confidence": float(chosen_confidence),
            "model_used": model_used.lower(),
            "cascade_path": model_used.lower(),
            "total_time_ms": inference_time_ms,
            "probabilities": chosen_probs,
        }

        # Energy tracking (unchanged)
        energy_kwh = None
        co2_grams = None
        if request.track_energy and energy_tracker:
            try:
                energy_log = energy_tracker.log_cascade_inference(cascade_like_result, actual_label=None)
                energy_kwh = energy_log.get("energy_kwh")
                co2_grams = energy_log.get("co2_grams")
            except Exception as e:
                print(f"⚠️ Energy tracking failed: {e}")

        # Database logging (unchanged) - include cascade_like_result fields
        if DB_AVAILABLE and db is not None:
            try:
                log_data = {
                    "email_id": request.email_id or f"api_{int(time.time() * 1000)}",
                    "email_text_preview": request.text[:200],
                    "predicted_class": cascade_like_result["prediction"],
                    "confidence": cascade_like_result["confidence"],
                    "model_used": cascade_like_result["model_used"],
                    "energy_kwh": energy_kwh or 0.0,
                    "co2_grams": co2_grams or 0.0,
                    "inference_time_ms": cascade_like_result.get("total_time_ms", 0),
                    "cascade_path": cascade_like_result.get("cascade_path", ""),
                }
                insert_inference_log(db, log_data)
            except Exception as e:
                print(f"⚠️ DB write failed: {e}")

        # Return richer response with per-model confidences & per-model probs (for dashboard)
        response = ClassificationResponse(
            email_id=request.email_id,
            prediction=cascade_like_result["prediction"],
            confidence=float(cascade_like_result["confidence"]),
            probabilities={k: float(v) for k, v in cascade_like_result["probabilities"].items()},
            model_used=cascade_like_result["model_used"],
            cascade_path=cascade_like_result.get("cascade_path", ""),
            inference_time_ms=float(cascade_like_result.get("total_time_ms", 0)),
            energy_kwh=energy_kwh,
            co2_grams=co2_grams,
            timestamp=datetime.utcnow(),
            per_model_confidences=per_model_confidences or None,
            per_model_probs=per_model_probs or None,
        )

        return response


    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Classification error: {str(e)}")


# -----------------------------
# /classify/batch
# -----------------------------
@app.post("/classify/batch", tags=["Classification"])
async def classify_batch(request: BatchEmailRequest):
    if router is None:
        raise HTTPException(status_code=503, detail="Models/router not loaded")

    if len(request.emails) > 100:
        raise HTTPException(status_code=400, detail="Maximum 100 emails per batch")

    results: List[Dict[str, Any]] = []
    db = None
    if DB_AVAILABLE:
        db = SessionLocal()

    try:
        for email_req in request.emails:
            try:
                start_time = time.time()
                pred_idx, model_used, details = router.predict_single(email_req.text)
                inference_time_ms = (time.time() - start_time) * 1000
                prediction_label = label_from_index(model_used, int(pred_idx))

                # extract probs/conf
                probs = {}
                confidence = 0.0
                try:
                    probs_list = details.get("probs") if details else None
                    if probs_list is not None:
                        labels = Config.CLASSES
                        if model_used.lower() == "green" and green_model is not None:
                            labels = list(map(str, green_model.classes_))
                        elif model_used.lower() in ("medium", "med") and medium_model is not None:
                            labels = [str(v) for v in getattr(medium_model, "id2label", getattr(medium_model, "classes_", Config.CLASSES))]
                        elif model_used.lower() == "heavy" and heavy_model is not None:
                            labels = [str(v) for v in getattr(heavy_model, "id2label", getattr(heavy_model, "classes_", Config.CLASSES))]
                        if len(labels) == len(probs_list):
                            probs = {str(labels[i]): float(probs_list[i]) for i in range(len(probs_list))}
                        else:
                            probs = {str(i): float(p) for i, p in enumerate(probs_list)}
                        confidence = max(probs.values()) if probs else 0.0
                    else:
                        confidence = float(details.get("confidence", 0.0)) if details else 0.0
                except Exception:
                    confidence = float(details.get("confidence", 0.0)) if details else 0.0

                cascade_like_result = {
                    "prediction": prediction_label,
                    "confidence": float(confidence),
                    "model_used": model_used.lower(),
                    "cascade_path": model_used.lower(),
                    "total_time_ms": inference_time_ms,
                    "probabilities": probs,
                }

                energy_kwh = None
                co2_grams = None
                if email_req.track_energy and energy_tracker:
                    try:
                        energy_log = energy_tracker.log_cascade_inference(cascade_like_result, actual_label=None)
                        energy_kwh = energy_log.get("energy_kwh")
                        co2_grams = energy_log.get("co2_grams")
                    except Exception as e:
                        print(f"⚠️ Energy tracking failed (batch): {e}")

                if DB_AVAILABLE and db is not None:
                    try:
                        log_data = {
                            "email_id": email_req.email_id or f"batch_{int(time.time() * 1000)}",
                            "email_text_preview": email_req.text[:200],
                            "predicted_class": cascade_like_result["prediction"],
                            "confidence": cascade_like_result["confidence"],
                            "model_used": cascade_like_result["model_used"],
                            "energy_kwh": energy_kwh or 0.0,
                            "co2_grams": co2_grams or 0.0,
                            "inference_time_ms": cascade_like_result.get("total_time_ms", 0),
                            "cascade_path": cascade_like_result.get("cascade_path", ""),
                        }
                        insert_inference_log(db, log_data)
                    except Exception as e:
                        print(f"⚠️ DB write failed (batch): {e}")

                results.append({
                    "email_id": email_req.email_id,
                    "prediction": cascade_like_result["prediction"],
                    "confidence": float(cascade_like_result["confidence"]),
                    "model_used": cascade_like_result["model_used"],
                    "cascade_path": cascade_like_result.get("cascade_path"),
                    "inference_time_ms": float(cascade_like_result.get("total_time_ms", 0)),
                    "energy_kwh": energy_kwh,
                    "co2_grams": co2_grams,
                })

            except Exception as e:
                traceback.print_exc()
                results.append({"email_id": email_req.email_id, "error": str(e)})

        successful = len([r for r in results if "error" not in r])
        return {"total": len(request.emails), "successful": successful, "results": results}
    finally:
        if DB_AVAILABLE and db is not None:
            db.close()


@app.get("/stats", response_model=StatsResponse, tags=["Statistics"])
async def get_statistics():
    try:
        # Router stats not used anymore → remove fake zeros
        db_stats = {}
        energy_savings = {}

        if DB_AVAILABLE:
            db = SessionLocal()
            try:
                db_stats = get_model_statistics(db) or {}
                energy_savings = calculate_energy_savings(db) or {}
            finally:
                db.close()

        total = int(db_stats.get("total_inferences", 0))

        return StatsResponse(
            total_inferences=total,
            cascade_accuracy=None,
            model_distribution={
                "green": 0.0,   # optional: you can compute real %
                "medium": 0.0,
                "heavy": 0.0,
            },
            energy_saved_percent = (
                energy_savings.get("energy_saved_percent")
                if energy_savings else None
            ),

            co2_saved_grams = (
                energy_savings.get("cascade_co2_grams")
                or energy_savings.get("co2_saved_grams")
                if energy_savings else None
            ),

            avg_inference_time_ms=float(db_stats.get("avg_inference_time_ms", 0)),
        )
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Statistics error: {str(e)}")



# -----------------------------
# /stats/models
# -----------------------------
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
# Router-info endpoint (calibration + per-class thresholds)
# Priority:
# 1) data/models/cascade/green_ai_config.json
# 2) data/models/cascade/cascade_config.pkl (joblib)
# 3) fallback to scalar thresholds from Config (if router present use that)
# -------------------------------------------------------------------------
@app.get("/config/router-info", tags=["Configuration"])
async def get_router_info():
    # 1) Preferred: JSON config file
    cfg_path = Config.MODELS_DIR / "cascade" / "green_ai_config.json"
    try:
        if cfg_path.exists():
            with open(cfg_path, "r") as f:
                cfg = json.load(f)
            cfg.setdefault("classes", Config.CLASSES)
            cfg.setdefault("num_classes", len(cfg.get("classes", Config.CLASSES)))
            return cfg
    except Exception as e:
        print(f"⚠️ Could not load green_ai_config.json: {e}")

    # 2) Fallback: try cascade_config.pkl saved by older flows
    pkl_path = Config.MODELS_DIR / "cascade" / "cascade_config.pkl"
    try:
        if pkl_path.exists():
            cfg_loaded = joblib.load(pkl_path)
            out = {
                "temperature_lr": cfg_loaded.get("temperature_lr"),
                "temperature_med": cfg_loaded.get("temperature_med"),
                "thresholds_lr": cfg_loaded.get("thresholds_lr", {}),
                "thresholds_med": cfg_loaded.get("thresholds_med", {}),
                "num_classes": cfg_loaded.get("num_classes", len(Config.CLASSES)),
                "classes": cfg_loaded.get("classes", Config.CLASSES),
                "validation_stats": cfg_loaded.get("validation_stats"),
            }
            if isinstance(out["thresholds_lr"], (float, int)):
                out["thresholds_lr"] = {str(i): float(out["thresholds_lr"]) for i in range(out["num_classes"])}
            if isinstance(out["thresholds_med"], (float, int)):
                out["thresholds_med"] = {str(i): float(out["thresholds_med"]) for i in range(out["num_classes"])}
            return out
    except Exception as e:
        print(f"⚠️ Could not load cascade_config.pkl: {e}")

    # 3) As last resort, try to read simple fields from loaded router or fall back to Config
    if router is not None:
        try:
            # Router stores per-class thresholds as T1/T2
            out = {
                "temperature_lr": getattr(router, "T_lr", None),
                "temperature_med": getattr(router, "T_med", None),
                "thresholds_lr": {str(k): float(v) for k, v in getattr(router, "T1", {}).items()} if getattr(router, "T1", None) else {str(i): float(Config.GREEN_THRESHOLD) for i in range(len(Config.CLASSES))},
                "thresholds_med": {str(k): float(v) for k, v in getattr(router, "T2", {}).items()} if getattr(router, "T2", None) else {str(i): float(Config.MEDIUM_THRESHOLD) for i in range(len(Config.CLASSES))},
                "num_classes": getattr(router, "num_classes", len(Config.CLASSES)),
                "classes": Config.CLASSES,
                "validation_stats": None,
            }
            return out
        except Exception as e:
            print(f"⚠️ Could not extract thresholds from router: {e}")

    raise HTTPException(status_code=503, detail="Router configuration not available. Run fit step and save green_ai_config.json to data/models/cascade/.")


# -----------------------------
# /config (compact summary)
# -----------------------------
@app.get("/config", tags=["Configuration"])
async def get_configuration():
    """Return a compact configuration summary (keeps backward-compatible fields)."""
    if router is None:
        raise HTTPException(status_code=503, detail="Models/router not loaded")

    # load router-info if present
    router_info = None
    try:
        router_info_path = (Config.MODELS_DIR / "cascade" / "green_ai_config.json")
        if router_info_path.exists():
            with open(router_info_path, "r") as f:
                router_info = json.load(f)
    except Exception:
        router_info = None

    return {
        "green_threshold": float(Config.GREEN_THRESHOLD),
        "medium_threshold": float(Config.MEDIUM_THRESHOLD),
        "classes": Config.CLASSES,
        "model_info": {
            "green": {
                "type": "TF-IDF + Logistic Regression",
                "size_mb": green_model.estimate_size() if green_model else 0,
            },
            "medium": {
                "type": "DistilBERT",
                "size_mb": medium_model.estimate_size() if medium_model else 0,
                "max_length": getattr(medium_model, "max_length", None),
            },
            "heavy": {
                "type": "DeBERTa-v3",
                "size_mb": heavy_model.estimate_size() if heavy_model else 0,
                "max_length": getattr(heavy_model, "max_length", None),
            },
        },
        "router_present": bool(router_info),
    }


# -----------------------------
# /config/thresholds (legacy scalar update)
# -----------------------------
@app.post("/config/thresholds", tags=["Configuration"])
async def update_thresholds(payload: ThresholdsRequest = Body(...)):
    """
    Backwards-compatible endpoint to update scalar thresholds (kept for legacy UX).
    Note: prefer updating per-class thresholds via offline fit and saving green_ai_config.json.
    """
    global router
    if router is None:
        raise HTTPException(status_code=503, detail="Models/router not loaded")

    if payload.medium_threshold > payload.green_threshold:
        raise HTTPException(status_code=400, detail="Medium threshold cannot be higher than green threshold")

    # Build a minimal config using provided scalars and save as legacy cascade_config.pkl for compatibility
    try:
        num_classes = len(Config.CLASSES)
        cfg = {
            "temperature_lr": getattr(router, "T_lr", 1.0),
            "temperature_med": getattr(router, "T_med", 1.0),
            "thresholds_lr": {str(i): float(payload.green_threshold) for i in range(num_classes)},
            "thresholds_med": {str(i): float(payload.medium_threshold) for i in range(num_classes)},
            "num_classes": num_classes,
            "classes": Config.CLASSES,
            "validation_stats": None,
        }

        # Save green_ai_config.json (preferred)
        cfg_path = Config.MODELS_DIR / "cascade"
        cfg_path.mkdir(parents=True, exist_ok=True)
        with open(cfg_path / "green_ai_config.json", "w") as f:
            json.dump(cfg, f, indent=2)

        # Update in-memory router
        router = GreenRouter(cfg, green_model, medium_model, heavy_model)

        # Also save legacy cascade_config.pkl for backward compatibility
        try:
            joblib.dump({
                "green_threshold": float(payload.green_threshold),
                "medium_threshold": float(payload.medium_threshold),
                "thresholds_lr": cfg["thresholds_lr"],
                "thresholds_med": cfg["thresholds_med"],
                "num_classes": num_classes,
                "classes": Config.CLASSES,
                "validation_stats": None,
            }, cfg_path / "cascade_config.pkl")
        except Exception as e:
            print(f"⚠️ Could not save cascade_config.pkl: {e}")

        return {
            "message": "Thresholds updated (scalar) successfully",
            "old_thresholds": {"green": Config.GREEN_THRESHOLD, "medium": Config.MEDIUM_THRESHOLD},
            "new_thresholds": {"green": payload.green_threshold, "medium": payload.medium_threshold},
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Could not update thresholds: {e}")


# -----------------------------
# /models/info
# -----------------------------
@app.get("/models/info", tags=["Models"])
async def get_models_info():
    if router is None:
        raise HTTPException(status_code=503, detail="Models/router not loaded")

    green_classes = getattr(green_model, "classes_", None)
    if hasattr(green_classes, "tolist"):
        green_classes = green_classes.tolist()

    return {
        "green_model": {
            "type": "TF-IDF + Logistic Regression",
            "size_mb": green_model.estimate_size() if green_model else 0,
            "features": len(green_model.feature_names) if green_model and getattr(green_model, "feature_names", None) is not None else 0,
            "classes": green_classes,
        },
        "medium_model": {
            "type": "DistilBERT",
            "size_mb": medium_model.estimate_size() if medium_model else 0,
            "max_length": getattr(medium_model, "max_length", None),
            "classes": getattr(medium_model, "classes_", None),
        },
        "heavy_model": {
            "type": "DeBERTa-v3-base",
            "size_mb": heavy_model.estimate_size() if heavy_model else 0,
            "max_length": getattr(heavy_model, "max_length", None),
            "classes": getattr(heavy_model, "classes_", None),
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
