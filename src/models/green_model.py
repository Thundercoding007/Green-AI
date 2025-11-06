# File: src/models/green_model.py
# Green Model: TF-IDF + Logistic Regression (Lightweight baseline)

import joblib
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, accuracy_score
import time
import pandas as pd
from src.config import Config


# ===============================================================
# MODEL CLASS
# ===============================================================
class GreenModel:
    """Lightweight email classifier using TF-IDF + Logistic Regression"""

    def __init__(self, max_features: int = 5000, random_state: int = 42):
        self.max_features = max_features
        self.random_state = random_state

        # TF-IDF Vectorizer
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),
            min_df=3,
            max_df=0.85,
            strip_accents="unicode",
            lowercase=True,
            stop_words="english",
            token_pattern=r"(?u)\b\w\w+\b",
        )

        # Base Logistic Regression
        self.classifier = LogisticRegression(
            C=0.1,
            max_iter=1000,
            random_state=random_state,
            class_weight="balanced",
            solver="liblinear",
            penalty="l2",
            multi_class="ovr",
        )

        self.calibrated_classifier = None
        self.classes_ = None
        self.is_trained = False
        self.training_time = None
        self.feature_names = None

    # -----------------------------------------------------------
    def train(self, X_train, y_train, X_val=None, y_val=None, calibrate=True):
        """Train the Green Model"""
        print("=" * 60)
        print("🌱 Training Green Model (TF-IDF + Logistic Regression)")
        print("=" * 60)

        start = time.time()

        # Fit TF-IDF
        X_train_tfidf = self.vectorizer.fit_transform(X_train)
        self.feature_names = self.vectorizer.get_feature_names_out()
        print(f"✅ TF-IDF features extracted: {len(self.feature_names)}")

        # Train base classifier
        self.classifier.fit(X_train_tfidf, y_train)
        self.classes_ = self.classifier.classes_

        # Calibrate probabilities (optional)
        if calibrate and X_val is not None and y_val is not None:
            print("🎚 Calibrating probabilities...")
            X_val_tfidf = self.vectorizer.transform(X_val)
            self.calibrated_classifier = CalibratedClassifierCV(
                self.classifier, method="sigmoid", cv="prefit"
            )
            self.calibrated_classifier.fit(X_val_tfidf, y_val)
        else:
            self.calibrated_classifier = self.classifier

        self.training_time = time.time() - start
        self.is_trained = True
        print(f"⏱ Training completed in {self.training_time:.2f} s")

        # Evaluate
        train_preds = self.predict(X_train)
        train_acc = accuracy_score(y_train, train_preds)
        print(f"📈 Train Accuracy: {train_acc:.4f}")

        val_acc = None
        if X_val is not None and y_val is not None:
            val_preds = self.predict(X_val)
            val_acc = accuracy_score(y_val, val_preds)
            print(f"📈 Validation Accuracy: {val_acc:.4f}")
            print("\n📊 Classification Report:")
            print(classification_report(y_val, val_preds, target_names=self.classes_))

        metrics = {
            "train_accuracy": train_acc,
            "val_accuracy": val_acc,
            "training_time": self.training_time,
            "num_features": len(self.feature_names),
            "model_size_mb": self.estimate_size(),
        }

        print(f"💾 Model size: ~{metrics['model_size_mb']:.2f} MB")
        return metrics

    # -----------------------------------------------------------
    def predict(self, X):
        """Predict class labels"""
        if not self.is_trained:
            raise ValueError("Model not trained yet.")
        X_tfidf = self.vectorizer.transform(X)
        return self.calibrated_classifier.predict(X_tfidf)

    def predict_proba(self, X):
        """Return class probability distributions"""
        if not self.is_trained:
            raise ValueError("Model not trained yet.")
        if isinstance(X, str):
            X = [X]
        X_tfidf = self.vectorizer.transform(X)
        return self.calibrated_classifier.predict_proba(X_tfidf)

    def predict_single(self, text: str):
        """Predict a single email with confidence and timing"""
        if not self.is_trained:
            raise ValueError("Model not trained yet.")
        start = time.time()
        proba = self.predict_proba([text])[0]
        pred_idx = np.argmax(proba)
        pred_class = self.classes_[pred_idx]
        return {
            "prediction": pred_class,
            "confidence": float(proba[pred_idx]),
            "inference_time_ms": (time.time() - start) * 1000,
            "probabilities": proba.tolist(),
        }

    # -----------------------------------------------------------
    def estimate_size(self):
        """Estimate approximate model size (MB)"""
        vocab_size = len(self.feature_names) if self.feature_names is not None else 0
        n_classes = len(self.classes_) if self.classes_ is not None else 2
        vocab_mb = (vocab_size * 100) / (1024 * 1024)
        weights_mb = (vocab_size * n_classes * 8) / (1024 * 1024)
        return vocab_mb + weights_mb

    # -----------------------------------------------------------
    def save(self, path: Path):
        """Save model and metadata"""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.vectorizer, path / "vectorizer.pkl")
        joblib.dump(self.calibrated_classifier, path / "classifier.pkl")
        metadata = {
            "classes": self.classes_.tolist(),
            "max_features": self.max_features,
            "training_time": self.training_time,
            "is_trained": self.is_trained,
        }
        joblib.dump(metadata, path / "metadata.pkl")
        print(f"✅ Green Model saved to: {path}")

    @classmethod
    def load(cls, path: Path):
        """Load model and metadata"""
        path = Path(path)
        metadata = joblib.load(path / "metadata.pkl")
        model = cls(max_features=metadata["max_features"])
        model.vectorizer = joblib.load(path / "vectorizer.pkl")
        model.calibrated_classifier = joblib.load(path / "classifier.pkl")
        model.classes_ = np.array(metadata["classes"])
        model.is_trained = metadata["is_trained"]
        model.training_time = metadata["training_time"]
        model.feature_names = model.vectorizer.get_feature_names_out()
        print(f"✅ Green Model loaded from: {path}")
        return model


# ===============================================================
# ENTRY POINT (VS Code / Colab)
# ===============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("🌿 Training Green Model")
    print("=" * 60)

    train_path = Config.PROCESSED_DATA_DIR / "train.csv"
    val_path = Config.PROCESSED_DATA_DIR / "val.csv"

    if not train_path.exists() or not val_path.exists():
        raise SystemExit("❌ Processed train/val CSVs not found. Run preprocessing first.")

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)

    model = GreenModel()
    metrics = model.train(
        X_train=train_df["processed_text"],
        y_train=train_df["label"],
        X_val=val_df["processed_text"],
        y_val=val_df["label"],
        calibrate=True,
    )

    model.save(Config.GREEN_MODEL_PATH)
    print("\n✅ Green Model training complete!")
