# File: src/models/cascade.py
# Cascade Classifier - Intelligent routing between models

import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import joblib
import json

from src.models.green_model import GreenModel
from src.models.medium_model import MediumModel
from src.models.heavy_model import HeavyModel
from src.config import Config


class CascadeClassifier:
    """
    Intelligent cascade classifier that routes emails through models
    based on confidence thresholds to minimize energy consumption.
    """

    def __init__(
        self,
        green_model: GreenModel,
        medium_model: MediumModel,
        heavy_model: HeavyModel,
        green_threshold: float = 0.65,
        medium_threshold: float = 0.60,
    ):
        """Initialize cascade classifier"""

        self.green_model = green_model
        self.medium_model = medium_model
        self.heavy_model = heavy_model

        # -------------------------------------------------------------
        # ✅ Auto-load optimized thresholds if available
        # -------------------------------------------------------------
        opt_path = Path("optimization_results/best_thresholds.json")
        if opt_path.exists():
            try:
                with open(opt_path, "r") as f:
                    best = json.load(f)
                self.green_threshold = best.get("green_threshold", green_threshold)
                self.medium_threshold = best.get("medium_threshold", medium_threshold)
                print(
                    f"✅ Loaded optimized thresholds: "
                    f"green={self.green_threshold:.2f}, medium={self.medium_threshold:.2f}"
                )
            except Exception as e:
                print(f"⚠️  Failed to load optimized thresholds ({e}) — using defaults.")
                self.green_threshold = green_threshold
                self.medium_threshold = medium_threshold
        else:
            print("⚠️  Optimization file not found — using default thresholds.")
            self.green_threshold = green_threshold
            self.medium_threshold = medium_threshold

        # Statistics tracking
        self.stats = {
            "total_inferences": 0,
            "green_used": 0,
            "medium_used": 0,
            "heavy_used": 0,
            "green_correct": 0,
            "medium_correct": 0,
            "heavy_correct": 0,
        }

    # ---------------------------------------------------------------------
    def predict_single(self, text: str, return_details: bool = True) -> Dict:
        """Predict class for a single email using cascade logic."""
        cascade_start = time.time()
        cascade_path = []

        # Tier 1 — Green Model
        green_start = time.time()
        green_result = self.green_model.predict_single(text)
        green_time = time.time() - green_start
        cascade_path.append("green")

        if "probabilities" not in green_result:
            proba = self.green_model.predict_proba([text])[0]
            green_result["probabilities"] = {
                self.green_model.classes_[i]: float(p) for i, p in enumerate(proba)
            }
        print(f"[DEBUG] Green conf={green_result['confidence']:.3f} | Threshold={self.green_threshold}")
        if green_result["confidence"] >= self.green_threshold:
            total_time = (time.time() - cascade_start) * 1000
            self.stats["green_used"] += 1
            self.stats["total_inferences"] += 1

            return {
                "prediction": green_result["prediction"],
                "confidence": green_result["confidence"],
                "model_used": "green",
                "cascade_path": "->".join(cascade_path),
                "total_time_ms": total_time,
                **({"details": green_result} if return_details else {}),
            }

        # Tier 2 — Medium Model
        medium_start = time.time()
        medium_result = self.medium_model.predict_single(text)
        medium_time = time.time() - medium_start
        cascade_path.append("medium")

        if "probabilities" not in medium_result:
            probs = self.medium_model.predict_proba([text])[0]
            medium_result["probabilities"] = {
                self.medium_model.id2label[i]: float(p) for i, p in enumerate(probs)
            }
        
        if medium_result["confidence"] >= self.medium_threshold:
            total_time = (time.time() - cascade_start) * 1000
            self.stats["medium_used"] += 1
            self.stats["total_inferences"] += 1

            return {
                "prediction": medium_result["prediction"],
                "confidence": medium_result["confidence"],
                "model_used": "medium",
                "cascade_path": "->".join(cascade_path),
                "total_time_ms": total_time,
                **({"details": medium_result} if return_details else {}),
            }

        # Tier 3 — Heavy Model (fallback)
        heavy_start = time.time()
        heavy_result = self.heavy_model.predict_single(text)
        heavy_time = time.time() - heavy_start
        cascade_path.append("heavy")
        total_time = (time.time() - cascade_start) * 1000

        self.stats["heavy_used"] += 1
        self.stats["total_inferences"] += 1

        return {
            "prediction": heavy_result["prediction"],
            "confidence": heavy_result["confidence"],
            "model_used": "heavy",
            "cascade_path": "->".join(cascade_path),
            "total_time_ms": total_time,
            **({"details": heavy_result} if return_details else {}),
        }

    # ---------------------------------------------------------------------
    def predict_batch(
        self,
        texts: List[str],
        y_true: Optional[List[str]] = None,
        track_accuracy: bool = True,
    ) -> List[Dict]:
        """Predict for multiple emails and optionally track accuracy."""
        results = []
        for i, text in enumerate(texts):
            result = self.predict_single(text)
            if y_true is not None and track_accuracy:
                correct = result["prediction"] == y_true[i]
                result["correct"] = correct
                key = f"{result['model_used']}_correct"
                if correct:
                    self.stats[key] += 1
            self.stats["total_inferences"] += 1
            results.append(result)
        return results

    # ---------------------------------------------------------------------
    def evaluate(self, X_test: List[str], y_test: List[str]) -> Dict:
        """Evaluate cascade performance."""
        from sklearn.metrics import accuracy_score, classification_report

        print("\n🎯 Evaluating Cascade Classifier...")
        self.stats.update({k: 0 for k in self.stats})  # Reset stats

        results = self.predict_batch(X_test, y_test)
        y_pred = [r["prediction"] for r in results]
        acc = accuracy_score(y_test, y_pred)

        avg_time = np.mean([r["total_time_ms"] for r in results])
        total = self.stats["total_inferences"] or 1
        green_pct = (self.stats["green_used"] / total) * 100
        medium_pct = (self.stats["medium_used"] / total) * 100
        heavy_pct = (self.stats["heavy_used"] / total) * 100

        print(f"   Accuracy: {acc*100:.2f}%")
        print(f"   Avg Inference Time: {avg_time:.2f} ms")
        print(f"   Green Used:  {green_pct:.1f}%")
        print(f"   Medium Used: {medium_pct:.1f}%")
        print(f"   Heavy Used:  {heavy_pct:.1f}%")
        print("\n📊 Classification Report:")
        print(classification_report(y_test, y_pred))

        return {
            "accuracy": acc,
            "avg_inference_time_ms": avg_time,
            "green_usage_pct": green_pct,
            "medium_usage_pct": medium_pct,
            "heavy_usage_pct": heavy_pct,
            "stats": self.stats,
        }
    def get_statistics(self) -> dict:
        """Return model usage statistics with accurate percentages."""
        total = (
            self.stats.get("green_used", 0)
            + self.stats.get("medium_used", 0)
            + self.stats.get("heavy_used", 0)
        )
        total = max(total, 1)  # avoid division by zero

        return {
            "green_used": self.stats.get("green_used", 0),
            "medium_used": self.stats.get("medium_used", 0),
            "heavy_used": self.stats.get("heavy_used", 0),
            "green_usage_pct": (self.stats.get("green_used", 0) / total) * 100,
            "medium_usage_pct": (self.stats.get("medium_used", 0) / total) * 100,
            "heavy_usage_pct": (self.stats.get("heavy_used", 0) / total) * 100,
            "total_inferences": total,
        }


    # ---------------------------------------------------------------------
    def save(self, path: Path):
        """Save cascade configuration."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        config = {
            "green_threshold": self.green_threshold,
            "medium_threshold": self.medium_threshold,
            "stats": self.stats,
        }
        joblib.dump(config, path / "cascade_config.pkl")
        print(f"✅ Cascade configuration saved at: {path}")

    # ---------------------------------------------------------------------
    @classmethod
    def load_models_and_create(
        cls,
        green_path: Path = Config.GREEN_MODEL_PATH,
        medium_path: Path = Config.MEDIUM_MODEL_PATH,
        heavy_path: Path = Config.HEAVY_MODEL_PATH,
        config_path: Optional[Path] = Config.MODELS_DIR,
    ):
        """Load all models and return a cascade classifier."""
        print("\n📦 Loading all models for Cascade Classifier...")
        green_model = GreenModel.load(green_path)
        medium_model = MediumModel.load(medium_path)
        heavy_model = HeavyModel.load(heavy_path)

        # Prefer cascade_config.pkl if exists
        if config_path and (config_path / "cascade_config.pkl").exists():
            cfg = joblib.load(config_path / "cascade_config.pkl")
            print(
                f"✅ Loaded thresholds: "
                f"green={cfg['green_threshold']:.2f}, medium={cfg['medium_threshold']:.2f}"
            )
            return cls(
                green_model,
                medium_model,
                heavy_model,
                cfg["green_threshold"],
                cfg["medium_threshold"],
            )

        # Else, rely on optimized file logic in __init__
        return cls(green_model, medium_model, heavy_model)


# ---------------------------------------------------------------------
# 🧪 Standalone test
# ---------------------------------------------------------------------
if __name__ == "__main__":
    import pandas as pd

    print("=" * 70)
    print("🌿 Testing Cascade Classifier")
    print("=" * 70)

    test_df = pd.read_csv(Config.PROCESSED_DATA_DIR / "test.csv")
    cascade = CascadeClassifier.load_models_and_create()

    print("\n🎯 Testing single prediction...")
    sample = test_df["processed_text"].iloc[0]
    result = cascade.predict_single(sample)
    print(
        f"Prediction: {result['prediction']}, "
        f"Model: {result['model_used']}, "
        f"Confidence: {result['confidence']:.2f}"
    )

    print("\n📊 Evaluating Cascade on small subset (n=300)...")
    subset = test_df.sample(n=300, random_state=42)
    metrics = cascade.evaluate(subset["processed_text"].tolist(), subset["label"].tolist())
    print("\n✅ Cascade evaluation complete!")
