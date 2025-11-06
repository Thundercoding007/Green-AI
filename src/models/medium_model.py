# File: src/models/medium_model.py
# Medium Model: DistilBERT (Balanced performance)

import os
os.environ["WANDB_DISABLED"] = "true"  # Disable Weights & Biases prompts

import torch
import numpy as np
import time
from pathlib import Path
from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import joblib
from torch.utils.data import Dataset


# ===============================================================
# DATASET CLASS
# ===============================================================
class EmailDataset(Dataset):
    """PyTorch Dataset for emails"""

    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(label, dtype=torch.long),
        }


# ===============================================================
# MODEL CLASS
# ===============================================================
class MediumModel:
    """Medium-weight email classifier using DistilBERT"""

    def __init__(self, num_classes: int = 3, max_length: int = 256, random_state: int = 42):
        self.num_classes = num_classes
        self.max_length = max_length
        self.random_state = random_state

        torch.manual_seed(random_state)
        np.random.seed(random_state)

        # Device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🖥️  Using device: {self.device}")

        # Initialize tokenizer and model placeholders
        self.tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        self.model = None
        self.trainer = None

        # Metadata
        self.classes_ = None
        self.label2id = None
        self.id2label = None
        self.is_trained = False
        self.training_time = None

    # -----------------------------------------------------------
    def _create_model(self):
        """Create DistilBERT model"""
        if self.label2id is None:
            raise ValueError("Label mappings not initialized. Run _prepare_data first.")

        self.model = DistilBertForSequenceClassification.from_pretrained(
            "distilbert-base-uncased",
            num_labels=len(self.label2id),
            id2label=self.id2label,
            label2id=self.label2id,
        ).to(self.device)

    # -----------------------------------------------------------
    def _prepare_data(self, X, y):
        """Prepare dataset"""
        if isinstance(y.iloc[0], str):
            if self.classes_ is None:
                self.classes_ = sorted(y.unique())
                self.label2id = {label: idx for idx, label in enumerate(self.classes_)}
                self.id2label = {idx: label for label, idx in self.label2id.items()}
            y_numeric = y.map(self.label2id).values
        else:
            y_numeric = y.values

        return EmailDataset(X.values, y_numeric, self.tokenizer, self.max_length)

    # -----------------------------------------------------------
    def train(self, X_train, y_train, X_val=None, y_val=None,
              batch_size=16, epochs=3, learning_rate=2e-5):
        """Train the DistilBERT model"""
        print("🔬 Training Medium Model (DistilBERT)...")
        print("-" * 60)

        start_time = time.time()

        train_dataset = self._prepare_data(X_train, y_train)
        val_dataset = self._prepare_data(X_val, y_val) if X_val is not None else None

        print(f"📊 Training samples: {len(train_dataset)}")
        if val_dataset:
            print(f"📊 Validation samples: {len(val_dataset)}")

        # Create model
        self._create_model()

        # Training configuration
        training_args = TrainingArguments(
            output_dir=str(Path("data/models/medium_checkpoints")),
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size * 2,
            learning_rate=learning_rate,
            weight_decay=0.01,
            warmup_steps=500,
            logging_steps=100,
            evaluation_strategy="epoch" if val_dataset else "no",
            save_strategy="epoch",
            load_best_model_at_end=bool(val_dataset),
            metric_for_best_model="accuracy",
            save_total_limit=2,
            seed=self.random_state,
            fp16=torch.cuda.is_available(),
            report_to="none"
        )

        # Metrics
        def compute_metrics(eval_pred):
            predictions, labels = eval_pred
            predictions = np.argmax(predictions, axis=1)
            accuracy = accuracy_score(labels, predictions)
            precision, recall, f1, _ = precision_recall_fscore_support(
                labels, predictions, average="weighted"
            )
            return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)] if val_dataset else [],
        )

        print(f"🚀 Starting training for {epochs} epochs...")
        self.trainer.train()

        self.training_time = time.time() - start_time
        self.is_trained = True
        print(f"\n✅ Training complete in {self.training_time / 60:.2f} minutes")

        if val_dataset:
            print("\n📈 Validation Performance:")
            results = self.trainer.evaluate()
            for k, v in results.items():
                if k.startswith("eval_"):
                    print(f"   {k.replace('eval_', '')}: {v:.4f}")

        metrics = {
            "training_time": self.training_time,
            "model_size_mb": self.estimate_size(),
            "num_parameters": sum(p.numel() for p in self.model.parameters()),
        }
        return metrics

    # -----------------------------------------------------------
    def predict_proba(self, texts):
        """Return probability distributions for one or more texts."""
        if not self.is_trained:
            raise ValueError("Model not trained or loaded.")

        if isinstance(texts, str):
            texts = [texts]

        self.model.eval()
        encodings = self.tokenizer(
            texts,
            add_special_tokens=True,
            max_length=self.max_length,
            truncation=True,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            logits = self.model(**encodings).logits
            probs = torch.softmax(logits, dim=1).cpu().numpy()

        return probs

    # -----------------------------------------------------------
    def predict_single(self, text: str):
        """Predict a single email with confidence and inference time"""
        if not self.is_trained:
            raise ValueError("Model not trained or loaded.")

        start = time.time()
        probs = self.predict_proba([text])[0]
        pred_idx = np.argmax(probs)
        pred_class = self.id2label[pred_idx]
        return {
            "prediction": pred_class,
            "confidence": float(probs[pred_idx]),
            "inference_time_ms": (time.time() - start) * 1000,
            "probabilities": probs.tolist(),
        }

    # -----------------------------------------------------------
    def save(self, path: Path):
        """Save model to disk"""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        joblib.dump({
            "classes": self.classes_,
            "label2id": self.label2id,
            "id2label": self.id2label,
            "max_length": self.max_length,
            "training_time": self.training_time,
            "is_trained": self.is_trained,
        }, path / "metadata.pkl")
        print(f"✅ Model saved at {path}")

    @classmethod
    def load(cls, path: Path):
        """Load model from disk"""
        path = Path(path)
        metadata = joblib.load(path / "metadata.pkl")
        model = cls(num_classes=len(metadata["classes"]), max_length=metadata["max_length"])
        model.tokenizer = DistilBertTokenizer.from_pretrained(path)
        model.model = DistilBertForSequenceClassification.from_pretrained(path).to(model.device)
        model.classes_ = metadata["classes"]
        model.label2id = metadata["label2id"]
        model.id2label = metadata["id2label"]
        model.training_time = metadata["training_time"]
        model.is_trained = metadata["is_trained"]
        print(f"✅ Model loaded from {path}")
        return model

    # -----------------------------------------------------------
    def estimate_size(self):
        if self.model is None:
            return 0
        return (sum(p.numel() for p in self.model.parameters()) * 4) / (1024 * 1024)


# ===============================================================
# ENTRY POINT
# ===============================================================
if __name__ == "__main__":
    import pandas as pd
    from src.config import Config

    print("=" * 60)
    print("🔬 Training Medium Model (DistilBERT)")
    print("=" * 60)

    train_df = pd.read_csv(Config.PROCESSED_DATA_DIR / "train.csv")
    val_df = pd.read_csv(Config.PROCESSED_DATA_DIR / "val.csv")

    model = MediumModel(num_classes=len(Config.CLASSES))

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
    print("\n✅ Training complete!")
    print(metrics)
