# File: src/models/heavy_model.py
# Heavy Model: DeBERTa-v3 (High accuracy, baseline)

import torch
import numpy as np
import time
from pathlib import Path
from transformers import (
    DebertaV2Tokenizer,
    DebertaV2ForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import joblib
from torch.utils.data import Dataset
from tqdm import tqdm


# ===============================================================
# DATASET CLASS
# ===============================================================
class EmailDataset(Dataset):
    """PyTorch Dataset for emails"""
    def __init__(self, texts, labels, tokenizer, max_length=512):
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
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(label, dtype=torch.long)
        }


# ===============================================================
# MODEL CLASS
# ===============================================================
class HeavyModel:
    """Heavy-weight email classifier using DeBERTa-v3"""

    def __init__(self, num_classes=3, max_length=512,
                 model_name="microsoft/deberta-v3-base", random_state=42):
        torch.manual_seed(random_state)
        np.random.seed(random_state)

        self.num_classes = num_classes
        self.max_length = max_length
        self.model_name = model_name
        self.random_state = random_state

        # Device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🖥️  Using device: {self.device}")
        if self.device.type == "cpu":
            print("⚠️  WARNING: Running on CPU will be very slow. Consider using GPU.")

        # Tokenizer
        self.tokenizer = DebertaV2Tokenizer.from_pretrained(model_name)

        # Model placeholders
        self.model = None
        self.trainer = None
        self.temperature = 1.0

        # Metadata
        self.classes_ = None
        self.label2id = None
        self.id2label = None
        self.is_trained = False
        self.training_time = None

    # -----------------------------------------------------------
    def _create_model(self):
        """Create DeBERTa model"""
        if not self.label2id:
            raise ValueError("Label mappings not initialized.")
        self.model = DebertaV2ForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=len(self.label2id),
            id2label=self.id2label,
            label2id=self.label2id
        ).to(self.device)

    def _prepare_data(self, X, y):
        """Prepare dataset and label encoding"""
        if isinstance(y.iloc[0], str):
            if self.classes_ is None:
                self.classes_ = sorted(y.unique())
                self.label2id = {label: i for i, label in enumerate(self.classes_)}
                self.id2label = {i: label for label, i in self.label2id.items()}
            y_numeric = y.map(self.label2id).values
        else:
            y_numeric = y.values
        return EmailDataset(X.values, y_numeric, self.tokenizer, self.max_length)

    # -----------------------------------------------------------
    def train(self, X_train, y_train, X_val=None, y_val=None,
              batch_size=8, epochs=4, learning_rate=1e-5):
        """Train the DeBERTa model"""
        print("=" * 60)
        print("🚀 Training Heavy Model (DeBERTa-v3-base)")
        print("=" * 60)

        start_time = time.time()
        train_dataset = self._prepare_data(X_train, y_train)
        val_dataset = self._prepare_data(X_val, y_val) if X_val is not None else None

        self._create_model()

        training_args = TrainingArguments(
            output_dir="./data/models/heavy_checkpoints",
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size * 2,
            learning_rate=learning_rate,
            weight_decay=0.01,
            warmup_ratio=0.1,
            logging_steps=100,
            evaluation_strategy="epoch" if val_dataset else "no",
            save_strategy="epoch",
            load_best_model_at_end=bool(val_dataset),
            metric_for_best_model="accuracy",
            save_total_limit=2,
            seed=self.random_state,
            fp16=torch.cuda.is_available(),
            gradient_accumulation_steps=2,
            dataloader_num_workers=0,
            report_to="none"
        )

        def compute_metrics(eval_pred):
            preds, labels = eval_pred
            preds = np.argmax(preds, axis=1)
            acc = accuracy_score(labels, preds)
            prec, rec, f1, _ = precision_recall_fscore_support(
                labels, preds, average="weighted", zero_division=0
            )
            return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}

        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)] if val_dataset else []
        )

        self.trainer.train()
        self.training_time = time.time() - start_time
        self.is_trained = True
        print(f"\n✅ Training completed in {self.training_time / 60:.2f} minutes")

        if val_dataset is not None:
            results = self.trainer.evaluate()
            print("\n📈 Validation Performance:")
            for k, v in results.items():
                if k.startswith("eval_"):
                    print(f"   {k.replace('eval_', '')}: {v:.4f}")

        return {
            "training_time": self.training_time,
            "model_size_mb": self.estimate_size(),
            "num_parameters": sum(p.numel() for p in self.model.parameters())
        }

    # -----------------------------------------------------------
    def predict_proba(self, texts):
        """Return softmax probabilities for one or more texts"""
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
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            logits = self.model(**encodings).logits / self.temperature
            probs = torch.softmax(logits, dim=1).cpu().numpy()

        return probs

    def predict_single(self, text):
        """Predict a single email with timing and confidence"""
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
            "probabilities": probs.tolist()
        }

    def predict(self, X):
        """Batch class prediction"""
        probs = self.predict_proba(X)
        preds = np.argmax(probs, axis=1)
        return np.array([self.id2label[i] for i in preds])

    # -----------------------------------------------------------
    def save(self, path: Path):
        """Save model + metadata"""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        joblib.dump({
            "classes": self.classes_,
            "label2id": self.label2id,
            "id2label": self.id2label,
            "max_length": self.max_length,
            "model_name": self.model_name,
            "training_time": self.training_time,
            "is_trained": self.is_trained,
            "temperature": self.temperature
        }, path / "metadata.pkl")
        print(f"✅ Heavy model saved to: {path}")

    @classmethod
    def load(cls, path: Path):
        """Load model + metadata"""
        path = Path(path)
        metadata = joblib.load(path / "metadata.pkl")

        model = cls(
            num_classes=len(metadata["classes"]),
            max_length=metadata["max_length"],
            model_name=metadata["model_name"]
        )
        model.tokenizer = DebertaV2Tokenizer.from_pretrained(path)
        model.model = DebertaV2ForSequenceClassification.from_pretrained(path).to(model.device)
        model.model.eval()

        model.classes_ = metadata["classes"]
        model.label2id = metadata["label2id"]
        model.id2label = metadata["id2label"]
        model.training_time = metadata["training_time"]
        model.is_trained = metadata["is_trained"]
        model.temperature = metadata.get("temperature", 1.0)
        print(f"✅ Heavy model loaded from: {path}")
        return model

    # -----------------------------------------------------------
    def estimate_size(self):
        """Estimate model size in MB"""
        if self.model is None:
            return 0
        num_params = sum(p.numel() for p in self.model.parameters())
        return (num_params * 4) / (1024 * 1024)


# ===============================================================
# ENTRY POINT
# ===============================================================
if __name__ == "__main__":
    import pandas as pd
    from src.config import Config

    print("=" * 60)
    print("🚀 Training Heavy Model (DeBERTa-v3-base)")
    print("=" * 60)

    train_df = pd.read_csv(Config.PROCESSED_DATA_DIR / "train.csv")
    val_df = pd.read_csv(Config.PROCESSED_DATA_DIR / "val.csv")

    model = HeavyModel(num_classes=len(Config.CLASSES))
    metrics = model.train(
        X_train=train_df["processed_text"],
        y_train=train_df["label"],
        X_val=val_df["processed_text"],
        y_val=val_df["label"],
        batch_size=8,
        epochs=4,
        learning_rate=1e-5
    )

    model.save(Config.HEAVY_MODEL_PATH)
    print("\n✅ Heavy model training complete!")
    print(metrics)
