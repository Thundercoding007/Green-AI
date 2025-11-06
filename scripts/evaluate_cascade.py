# File: scripts/evaluate_cascade.py
# ✅ Unified and Improved Evaluation Script for Cascade vs Baseline
# Works seamlessly in VS Code and Colab

import sys
import time
import json
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix

# ---------------------------------------------------------------------
# ✅ Ensure project root is in sys.path
# ---------------------------------------------------------------------
sys.path.append(str(Path(__file__).parent.parent))

from src.config import Config
from src.models.cascade import CascadeClassifier
from src.models.heavy_model import HeavyModel
from src.utils.energy_tracker import CascadeEnergyTracker

# Optional DB imports
try:
    from src.database import SessionLocal, insert_inference_log
    DB_AVAILABLE = True
except ImportError:
    DB_AVAILABLE = False
    print("⚠️  Database integration skipped (no src/database module found).")


# ---------------------------------------------------------------------
def evaluate_baseline(heavy_model, X_test, y_test, energy_tracker):
    """Evaluate baseline using only the heavy model."""
    print("\n🧠 Evaluating Baseline (Heavy Model Only)...")
    predictions = []

    for i, text in enumerate(tqdm(X_test, desc="Baseline inference", ncols=80)):
        result = heavy_model.predict_single(text)
        predictions.append(result["prediction"])

        # Log baseline inference energy
        energy_tracker.log_baseline_inference(
            prediction=result["prediction"],
            confidence=result["confidence"],
            inference_time_ms=result.get("inference_time_ms", 0),
            actual_label=y_test[i],
        )

    acc = accuracy_score(y_test, predictions)
    print(f"✅ Baseline Accuracy: {acc * 100:.2f}%")
    return {"accuracy": acc, "predictions": predictions}


# ---------------------------------------------------------------------
def evaluate_cascade(cascade, X_test, y_test, energy_tracker):
    """Evaluate the multi-tier cascade classifier."""
    print("\n🌿 Evaluating Cascade Classifier...")
    predictions = []

    for i, text in enumerate(tqdm(X_test, desc="Cascade inference", ncols=80)):
        result = cascade.predict_single(text, return_details=True)
        predictions.append(result["prediction"])
        energy_tracker.log_cascade_inference(result, actual_label=y_test[i])

    acc = accuracy_score(y_test, predictions)

    # Handle missing get_statistics() gracefully
    if hasattr(cascade, "get_statistics"):
        stats = cascade.get_statistics()
    else:
        stats = {"green_usage_pct": 0, "medium_usage_pct": 0, "heavy_usage_pct": 0}

    print(f"✅ Cascade Accuracy: {acc * 100:.2f}%")
    print(f"\n📈 Model Usage:")
    print(f"   Green:  {stats['green_usage_pct']:5.1f}%")
    print(f"   Medium: {stats['medium_usage_pct']:5.1f}%")
    print(f"   Heavy:  {stats['heavy_usage_pct']:5.1f}%")

    return {
        "accuracy": acc,
        "predictions": predictions,
        "stats": stats,
        "green_threshold": getattr(cascade, "green_threshold", 0.85),
        "medium_threshold": getattr(cascade, "medium_threshold", 0.80),
    }


# ---------------------------------------------------------------------
def save_to_database(cascade_logs):
    """Save cascade inference logs to the database."""
    if not DB_AVAILABLE:
        print("⚠️  Skipping database save (DB unavailable).")
        return

    print("\n💾 Saving inference logs to database...")
    db = SessionLocal()
    try:
        for i, log in enumerate(cascade_logs):
            insert_inference_log(db, {
                "email_id": f"eval_{i}",
                "predicted_class": log["prediction"],
                "actual_class": log.get("actual_label"),
                "confidence": log["confidence"],
                "correct": log.get("correct"),
                "model_used": log["model_used"],
                "energy_kwh": log["energy_kwh"],
                "co2_grams": log["co2_grams"],
                "inference_time_ms": log["inference_time_ms"],
                "cascade_path": log.get("cascade_path"),
            })
        print(f"✅ Saved {len(cascade_logs)} logs to DB.")
    except Exception as e:
        print(f"⚠️  Database error: {e}")
    finally:
        db.close()


# ---------------------------------------------------------------------
def main():
    """Main evaluation entry point."""
    print("=" * 70)
    print("🌿 GreenAI Email Classifier — Cascade Evaluation")
    print("=" * 70)

    # -----------------------------
    # Step 1: Load test dataset
    # -----------------------------
    print("\n📂 Loading test data...")
    test_path = Config.PROCESSED_DATA_DIR / "test.csv"
    if not test_path.exists():
        raise FileNotFoundError(f"❌ Test file not found: {test_path}")

    test_df = pd.read_csv(test_path)
    print(f"✅ Loaded {len(test_df)} samples.")

    # Limit sample size for faster testing
    sample_size = min(1000, len(test_df))
    test_df = test_df.sample(n=sample_size, random_state=42)
    print(f"Using {sample_size} samples for evaluation.\n")

    X_test = test_df["processed_text"].tolist()
    y_test = test_df["label"].tolist()

    # -----------------------------
    # Step 2: Initialize tracker
    # -----------------------------
    print("⚡ Initializing Energy Tracker...")
    energy_tracker = CascadeEnergyTracker(output_dir=Config.EMISSIONS_DIR)

    # -----------------------------
    # Step 3: Load models
    # -----------------------------
    print("\n📦 Loading Models...")
    cascade = CascadeClassifier.load_models_and_create(
        Config.GREEN_MODEL_PATH,
        Config.MEDIUM_MODEL_PATH,
        Config.HEAVY_MODEL_PATH,
    )
    heavy_model = HeavyModel.load(Config.HEAVY_MODEL_PATH)

    # -----------------------------
    # Step 4: Evaluate
    # -----------------------------
    baseline_metrics = evaluate_baseline(heavy_model, X_test, y_test, energy_tracker)
    cascade_metrics = evaluate_cascade(cascade, X_test, y_test, energy_tracker)
    cascade_metrics["y_true"] = y_test

    # -----------------------------
    # Step 5: Compute savings
    # -----------------------------
    print("\n💰 Calculating Energy & CO₂ Savings...")
    energy_savings = energy_tracker.calculate_savings()
    energy_savings["model_distribution"] = energy_tracker.get_model_distribution()

    # -----------------------------
    # Step 6: Display summary
    # -----------------------------
    print("\n📊 Final Results:")
    print(f"   Cascade Accuracy:  {cascade_metrics['accuracy'] * 100:.2f}%")
    print(f"   Baseline Accuracy: {baseline_metrics['accuracy'] * 100:.2f}%")
    print(f"   Energy Savings:    {energy_savings['energy_saved_percent']:.2f}%")
    print(f"   CO₂ Savings:       {energy_savings['co2_saved_percent']:.2f}%")

    # -----------------------------
    # Step 7: Save outputs
    # -----------------------------
    output_dir = Config.PROJECT_ROOT / "evaluation_results"
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "cascade": cascade_metrics,
        "baseline": baseline_metrics,
        "energy": energy_savings,
        "timestamp": str(pd.Timestamp.now())
    }

    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"💾 Saved results: {output_dir / 'results.json'}")

    energy_tracker.save_logs(output_dir)
    save_to_database(energy_tracker.cascade_logs)

    print("\n✅ Evaluation complete!")
    print(f"Results available in: {output_dir}")


# ---------------------------------------------------------------------
if __name__ == "__main__":
    main()
