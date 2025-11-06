#!/usr/bin/env python3
# File: scripts/train_all_models.py
# Complete end-to-end training pipeline for all three models
# Green (TF-IDF + Logistic Regression)
# Medium (DistilBERT)
# Heavy (DeBERTa-v3)

import sys
import time
import json
from pathlib import Path
import pandas as pd
import torch

# ---------------------------------------------------------------------
# ✅ 1. Path Setup
# ---------------------------------------------------------------------
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
sys.path.append(str(PROJECT_ROOT))

from src.config import Config
from src.models.green_model import GreenModel
from src.models.medium_model import MediumModel
from src.models.heavy_model import HeavyModel


# ---------------------------------------------------------------------
# ✅ Utility Functions
# ---------------------------------------------------------------------
def print_header(title: str):
    """Print a formatted header block"""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")


# ---------------------------------------------------------------------
# ✅ Training Functions
# ---------------------------------------------------------------------
def train_green_model(train_df, val_df):
    """Train the lightweight Green Model"""
    print_header("🌱 Step 1/3: Training Green Model (TF-IDF + Logistic Regression)")

    start = time.time()
    model = GreenModel(max_features=5000, random_state=Config.RANDOM_STATE)

    metrics = model.train(
        X_train=train_df["processed_text"],
        y_train=train_df["label"],
        X_val=val_df["processed_text"],
        y_val=val_df["label"],
        calibrate=True,
    )

    model.save(Config.GREEN_MODEL_PATH)
    metrics["total_time"] = time.time() - start

    print(f"\n✅ Green Model Complete!")
    print(f"   Time: {metrics['total_time']/60:.2f} minutes")
    print(f"   Accuracy: {metrics.get('val_accuracy', 'N/A')}")
    return metrics


def train_medium_model(train_df, val_df, use_subset=False):
    """Train the medium DistilBERT model"""
    print_header("🔬 Step 2/3: Training Medium Model (DistilBERT)")

    if use_subset:
        print("⚠️  Using 20% subset for faster training")
        train_df = train_df.sample(frac=0.2, random_state=42)
        val_df = val_df.sample(frac=0.2, random_state=42)

    start = time.time()
    model = MediumModel(
        num_classes=len(Config.CLASSES),
        max_length=256,
        random_state=Config.RANDOM_STATE,
    )
    print(f"🖥️  Device in use: {getattr(model, 'device', 'CPU (sklearn)')}")

    # ✅ Use Config path for checkpoints
    model_checkpoint_dir = Config.MEDIUM_MODEL_PATH / "checkpoints"

    metrics = model.train(
        X_train=train_df["processed_text"],
        y_train=train_df["label"],
        X_val=val_df["processed_text"],
        y_val=val_df["label"],
        batch_size=16,
        epochs=3,
        learning_rate=2e-5,
    )

    model.save(Config.MEDIUM_MODEL_PATH)
    metrics["total_time"] = time.time() - start

    print(f"\n✅ Medium Model Complete!")
    print(f"   Time: {metrics['total_time']/60:.2f} minutes")
    print(f"   Accuracy: {metrics.get('val_accuracy', 'N/A')}")
    return metrics


def train_heavy_model(train_df, val_df, use_subset=False):
    """Train the heavy DeBERTa-v3 model"""
    print_header("🚀 Step 3/3: Training Heavy Model (DeBERTa-v3)")

    if use_subset:
        print("⚠️  Using 20% subset for faster training")
        train_df = train_df.sample(frac=0.2, random_state=42)
        val_df = val_df.sample(frac=0.2, random_state=42)

    start = time.time()
    model = HeavyModel(
        num_classes=len(Config.CLASSES),
        max_length=512,
        random_state=Config.RANDOM_STATE,
    )
    print(f"🖥️  Device in use: {getattr(model, 'device', 'CPU (sklearn)')}")

    # ✅ Use Config path for checkpoints
    model_checkpoint_dir = Config.HEAVY_MODEL_PATH / "checkpoints"

    metrics = model.train(
        X_train=train_df["processed_text"],
        y_train=train_df["label"],
        X_val=val_df["processed_text"],
        y_val=val_df["label"],
        batch_size=8,
        epochs=4,
        learning_rate=1e-5,
    )

    model.save(Config.HEAVY_MODEL_PATH)
    metrics["total_time"] = time.time() - start

    print(f"\n✅ Heavy Model Complete!")
    print(f"   Time: {metrics['total_time']/60:.2f} minutes")
    print(f"   Accuracy: {metrics.get('val_accuracy', 'N/A')}")
    return metrics


# ---------------------------------------------------------------------
# ✅ Comparison Function
# ---------------------------------------------------------------------
def compare_models(test_df, metrics_summary):
    """Compare all trained models on the test set"""
    print_header("📊 Model Comparison on Test Set")

    from sklearn.metrics import accuracy_score

    results = {}
    print("Loading trained models...\n")

    green_model = GreenModel.load(Config.GREEN_MODEL_PATH)
    medium_model = MediumModel.load(Config.MEDIUM_MODEL_PATH)
    heavy_model = HeavyModel.load(Config.HEAVY_MODEL_PATH)

    models = {
        "Green (TF-IDF+LR)": green_model,
        "Medium (DistilBERT)": medium_model,
        "Heavy (DeBERTa)": heavy_model,
    }

    X_test, y_test = test_df["processed_text"], test_df["label"]

    for name, model in models.items():
        try:
            print(f"📈 Evaluating {name} ...")
            start = time.time()
            preds = model.predict(X_test)
            elapsed = time.time() - start

            acc = accuracy_score(y_test, preds)
            avg_ms = (elapsed / len(X_test)) * 1000
            size_mb = model.estimate_size()

            print(f"   Accuracy: {acc:.4f}")
            print(f"   Avg Inference: {avg_ms:.2f} ms")
            print(f"   Model Size: {size_mb:.2f} MB\n")

            results[name] = {
                "accuracy": acc,
                "avg_inference_ms": avg_ms,
                "model_size_mb": size_mb,
                "training_time_min": metrics_summary[name.split()[0].lower()]["total_time"] / 60,
            }
        except Exception as e:
            print(f"⚠️  {name} failed during evaluation: {e}")
            continue

    return results


# ---------------------------------------------------------------------
# ✅ Main Pipeline
# ---------------------------------------------------------------------
def main():
    print_header("🌿 GreenAI Email Classifier - Complete Training Pipeline")

    # ✅ Display Device Info
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Training will use: {device}\n")

    print("This will train:")
    print("  1️⃣  Green Model (TF-IDF + Logistic Regression)")
    print("  2️⃣  Medium Model (DistilBERT)")
    print("  3️⃣  Heavy Model (DeBERTa-v3)")
    print("\n⏱️  Estimated total time: 1–5 hours depending on hardware.\n")

    PROCEED = True
    USE_SUBSET = False  # ✅ Toggle this to True for quick tests

    if not PROCEED:
        print("❌ Training cancelled.")
        return

    # Load dataset splits
    print("📂 Loading processed datasets...")
    train_df = pd.read_csv(Config.PROCESSED_DATA_DIR / "train.csv")
    val_df = pd.read_csv(Config.PROCESSED_DATA_DIR / "val.csv")
    test_df = pd.read_csv(Config.PROCESSED_DATA_DIR / "test.csv")

    print(f"   Train: {len(train_df)}")
    print(f"   Val:   {len(val_df)}")
    print(f"   Test:  {len(test_df)}")

    overall_start = time.time()
    metrics_summary = {}

    try:
        # ✅ Skip already trained models to save time
        if not (Config.GREEN_MODEL_PATH / "metadata.pkl").exists():
            metrics_summary["green"] = train_green_model(train_df, val_df)
        else:
            print("✅ Green Model already trained — skipping retrain.\n")

        if not (Config.MEDIUM_MODEL_PATH / "metadata.pkl").exists():
            metrics_summary["medium"] = train_medium_model(train_df, val_df, USE_SUBSET)
        else:
            print("✅ Medium Model already trained — skipping retrain.\n")

        if not (Config.HEAVY_MODEL_PATH / "metadata.pkl").exists():
            metrics_summary["heavy"] = train_heavy_model(train_df, val_df, USE_SUBSET)
        else:
            print("✅ Heavy Model already trained — skipping retrain.\n")

        # Evaluate and compare
        comparison = compare_models(test_df, metrics_summary)
        total_time = time.time() - overall_start

        print_header("🎉 Training Complete!")
        print("📊 Final Summary:")
        print("-" * 70)
        print(f"{'Model':<25} {'Accuracy':<12} {'Inf. Time':<15} {'Size':<10}")
        print("-" * 70)

        for name, m in comparison.items():
            print(f"{name:<25} {m['accuracy']:<12.4f} "
                  f"{m['avg_inference_ms']:<15.2f}ms {m['model_size_mb']:<10.2f}MB")

        print("-" * 70)
        print(f"\n⏱️  Total training time: {total_time/3600:.2f} hours")

        # Save JSON summary
        summary_path = Config.MODELS_DIR / "training_summary.json"
        summary = {
            "training_summary": metrics_summary,
            "test_comparison": comparison,
            "total_training_time": total_time,
        }
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)

        print(f"\n💾 Summary saved to: {summary_path}")
        print("\n✅ All models trained successfully!")

    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user. Partial models saved.")
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()


# ---------------------------------------------------------------------
if __name__ == "__main__":
    main()
