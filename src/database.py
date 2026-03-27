# File: src/database.py
# Database Models and Connection Management

from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
from src.config import Config

# Create database engine
engine = create_engine(
    Config.DATABASE_URL,
    connect_args={"check_same_thread": False} if "sqlite" in Config.DATABASE_URL else {}
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class InferenceLog(Base):
    """Table to store each inference and its metrics"""
    __tablename__ = "inference_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    email_id = Column(String(100), index=True)
    email_text_preview = Column(String(200))  # First 200 chars for reference
    
    # Prediction details
    predicted_class = Column(String(50), index=True)
    actual_class = Column(String(50), nullable=True)  # If ground truth available
    confidence = Column(Float)
    correct = Column(Boolean, nullable=True)
    
    # Model used
    model_used = Column(String(20), index=True)  # 'green', 'medium', 'heavy'
    
    # Energy metrics
    energy_kwh = Column(Float)
    co2_grams = Column(Float)
    inference_time_ms = Column(Float)
    
    # Additional metadata
    cascade_path = Column(String(50))  # e.g., "green", "green->medium", "green->medium->heavy"


class ModelMetrics(Base):
    """Table to store aggregate model performance metrics"""
    __tablename__ = "model_metrics"
    
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    model_name = Column(String(20), index=True)  # 'green', 'medium', 'heavy', 'cascade'
    
    # Performance metrics
    accuracy = Column(Float)
    precision = Column(Float)
    recall = Column(Float)
    f1_score = Column(Float)
    
    # Usage statistics
    total_inferences = Column(Integer)
    usage_percentage = Column(Float)
    
    # Energy statistics
    total_energy_kwh = Column(Float)
    avg_energy_per_inference = Column(Float)
    total_co2_grams = Column(Float)
    
    # Timing
    avg_inference_time_ms = Column(Float)
    p95_inference_time_ms = Column(Float)
    p99_inference_time_ms = Column(Float)


class ThresholdConfiguration(Base):
    """Table to store cascade threshold configurations"""
    __tablename__ = "threshold_configurations"
    
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    green_threshold = Column(Float)
    medium_threshold = Column(Float)
    
    # Performance with these thresholds
    cascade_accuracy = Column(Float)
    energy_savings_percent = Column(Float)
    
    is_active = Column(Boolean, default=False)
    notes = Column(Text, nullable=True)


class BaselineComparison(Base):
    """Table to store baseline (heavy-only) comparison data"""
    __tablename__ = "baseline_comparison"
    
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    # Baseline metrics (heavy model only)
    baseline_accuracy = Column(Float)
    baseline_total_energy_kwh = Column(Float)
    baseline_total_co2_grams = Column(Float)
    baseline_avg_time_ms = Column(Float)
    
    # Cascade metrics
    cascade_accuracy = Column(Float)
    cascade_total_energy_kwh = Column(Float)
    cascade_total_co2_grams = Column(Float)
    cascade_avg_time_ms = Column(Float)
    
    # Savings
    energy_saved_percent = Column(Float)
    co2_saved_percent = Column(Float)
    accuracy_drop_percent = Column(Float)
    
    # Sample size
    num_samples = Column(Integer)


# Database utility functions
def init_db():
    """Initialize database tables"""
    Base.metadata.create_all(bind=engine)
    print("✅ Database tables created successfully!")


def get_db():
    """Get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def insert_inference_log(db, log_data: dict):
    """Insert a new inference log entry"""
    log = InferenceLog(**log_data)
    db.add(log)
    db.commit()
    db.refresh(log)
    return log


def get_model_statistics(db, model_name: str = None):
    """Get aggregated statistics for a model"""
    query = db.query(InferenceLog)
    
    if model_name:
        query = query.filter(InferenceLog.model_used == model_name)
    
    logs = query.all()
    
    if not logs:
        return None
    
    total = len(logs)
    correct = sum(1 for log in logs if log.correct)
    
    return {
    "total_inferences": total,
    "accuracy": correct / total if total > 0 else 0,

    # SAFE SUMS (None → 0)
    "total_energy_kwh": sum((log.energy_kwh or 0.0) for log in logs),
    "avg_energy_kwh": sum((log.energy_kwh or 0.0) for log in logs) / total,

    "total_co2_grams": sum((log.co2_grams or 0.0) for log in logs),
    "avg_co2_grams": sum((log.co2_grams or 0.0) for log in logs) / total,

    "avg_inference_time_ms": sum((log.inference_time_ms or 0.0) for log in logs) / total,
}



def calculate_energy_savings(db):
    """
    Compute realistic energy savings by comparing actual cascade usage
    vs a baseline where Heavy model runs 100% of the time.
    Works even if there are NO heavy-model logs in the DB.
    """

    logs = db.query(InferenceLog).all()
    if not logs:
        return {
            "energy_saved_percent": None,
            "cascade_co2_grams": None,
        }

    # ----------- Energy consumption (kWh) per model -----------
    ENERGY_GREEN = 0.00002   # 0.02 Wh
    ENERGY_MED   = 0.00015   # 0.15 Wh
    ENERGY_HEAVY = 0.00060   # 0.60 Wh

    # CO₂ intensity (grams per kWh)
    CO2_PER_KWH = 475.0

    # ----------- Compute baseline & actual energy -------------
    total_actual_kwh = 0.0
    total_baseline_kwh = 0.0

    for log in logs:
        # Baseline → Heavy would run for *every* inference
        total_baseline_kwh += ENERGY_HEAVY

        # Actual → depends on model used
        if log.model_used == "green":
            total_actual_kwh += ENERGY_GREEN
        elif log.model_used == "medium":
            total_actual_kwh += ENERGY_MED
        else:
            total_actual_kwh += ENERGY_HEAVY

    # ----------- Energy Saved ----------------
    energy_saved_kwh = max(0.0, total_baseline_kwh - total_actual_kwh)
    energy_saved_percent = (energy_saved_kwh / total_baseline_kwh) * 100.0

    # ----------- CO₂ Saved -------------------
    co2_saved_grams = energy_saved_kwh * CO2_PER_KWH

    return {
        "energy_saved_percent": round(energy_saved_percent, 4),
        "cascade_co2_grams": round(co2_saved_grams, 6),
        "actual_energy_kwh": round(total_actual_kwh, 6),
        "baseline_energy_kwh": round(total_baseline_kwh, 6)
    }


# Initialize database on module import
if __name__ == "__main__":
    init_db()