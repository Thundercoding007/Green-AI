# File: src/models/heavy_model.py
# Heavy Model: DeBERTa-v3 (High accuracy, baseline)
# Upgraded for low-GPU-VRAM environments:
#  - low_cpu_mem_usage on load
#  - batched inference with adaptive retry on OOM
#  - optional CPU fallback
#  - less memory fragmentation (empties cache between batches)
#  - minimal API changes (predict_proba/predict_single/predict unchanged)
#
# Author: Assistant (added resilience helpers)
# Note: model_name default remains "microsoft/deberta-v3-base"

import os
import torch
import numpy as np
import time
from pathlib import Path
from transformers import (
    DebertaV2Tokenizer,
    DebertaV2ForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import joblib
from torch.utils.data import Dataset
from tqdm import tqdm

# Optional environment tweak that can help fragmentation on some systems.
# The user can enable this in their environment if fragmentation is a problem.
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32,garbage_collection_threshold:0.6"

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
class HeavyModel:
    """Heavy-weight email classifier using DeBERTa-v3 (resilient to small GPU memory)"""

    def __init__(
        self,
        num_classes=3,
        max_length=512,
        model_name="microsoft/deberta-v3-base",
        random_state=42,
        prefer_cuda: bool = True,
    ):
        """
        prefer_cuda: if True and CUDA is available, will attempt to use CUDA.
        On OOM during inference, code will retry with smaller batches and finally fallback to CPU.
        """
        torch.manual_seed(random_state)
        np.random.seed(random_state)

        self.num_classes = num_classes
        self.max_length = max_length
        self.model_name = model_name
        self.random_state = random_state
        self.prefer_cuda = prefer_cuda

        # Device setup (respect prefer_cuda)
        if torch.cuda.is_available() and self.prefer_cuda:
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        print(f"🖥️  Using device: {self.device}")
        if self.device.type == "cpu":
            print("⚠️  WARNING: Running on CPU may be slow. The implementation will fallback to CPU on OOM.")

        # Tokenizer (light)
        self.tokenizer = DebertaV2Tokenizer.from_pretrained(self.model_name)

        # Model placeholders
        self.model = None
        self.trainer = None
        self.temperature = 1.0  # simple scalar temperature (float)

        # Metadata
        self.classes_ = None
        self.label2id = None
        self.id2label = None
        self.is_trained = False
        self.training_time = None

    # -----------------------------------------------------------
    def _create_model(self):
        """Create DeBERTa model with memory-conscious settings"""
        if not self.label2id:
            raise ValueError("Label mappings not initialized.")
        # Use low_cpu_mem_usage to reduce peak memory during from_pretrained
        map_location = None if self.device.type == "cuda" else "cpu"
        try:
            # load with low_cpu_mem_usage to avoid huge memory spikes on load
            self.model = DebertaV2ForSequenceClassification.from_pretrained(
                self.model_name,
                num_labels=len(self.label2id),
                id2label=self.id2label,
                label2id=self.label2id,
                low_cpu_mem_usage=True,
            )
            self.model.to(self.device)
        except Exception as e:
            # Fallback: load to CPU then move to device if possible
            print(f"⚠️ Model load with low_cpu_mem_usage failed ({e}), attempting safe CPU load...")
            self.model = DebertaV2ForSequenceClassification.from_pretrained(
                self.model_name,
                num_labels=len(self.label2id),
                id2label=self.id2label,
                label2id=self.label2id,
            )
            if self.device.type == "cuda":
                try:
                    self.model.to(self.device)
                except Exception as e2:
                    print(f"⚠️ Could not move model to CUDA ({e2}), staying on CPU.")
                    self.device = torch.device("cpu")
            else:
                self.model.to(self.device)

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
    def train(self, X_train, y_train, X_val=None, y_val=None, batch_size=8, epochs=4, learning_rate=1e-5):
        """Train the DeBERTa model (unchanged from before)"""
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
            per_device_eval_batch_size=max(1, batch_size // 2),
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
            fp16=torch.cuda.is_available() and self.device.type == "cuda",
            gradient_accumulation_steps=2,
            dataloader_num_workers=0,
            report_to="none",
        )

        def compute_metrics(eval_pred):
            preds, labels = eval_pred
            preds = np.argmax(preds, axis=1)
            acc = accuracy_score(labels, preds)
            prec, rec, f1, _ = precision_recall_fscore_support(labels, preds, average="weighted", zero_division=0)
            return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}

        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)] if val_dataset else [],
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
            "num_parameters": sum(p.numel() for p in self.model.parameters()),
        }

    # -----------------------------------------------------------
    def _batched_predict_proba(self, texts, batch_size=8):
        """
        Safe batched inference helper.
        - Tries to run on self.device with given batch_size.
        - On OOM reduces batch_size and retries.
        - If repeated OOMs, falls back to CPU and processes in small batches.
        Returns numpy array shape (N, C)
        """
        if isinstance(texts, str):
            texts = [texts]

        # normalize batch_size
        batch_size = max(1, int(batch_size))

        probs_list = []
        n = len(texts)

        # inner function to run one batch
        def run_batch(batch_texts):
            enc = self.tokenizer(
                batch_texts,
                add_special_tokens=True,
                max_length=self.max_length,
                truncation=True,
                padding=True,
                return_tensors="pt",
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}
            with torch.no_grad():
                outputs = self.model(**enc)
                logits = outputs.logits / (self.temperature if isinstance(self.temperature, (float, int)) else float(self.temperature))
                probs = torch.softmax(logits, dim=1).cpu().numpy()
            # free intermediate tensors
            if self.device.type == "cuda":
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            return probs

        # try progressive retries on OOM
        cur_batch = batch_size
        i = 0
        while i < n:
            end = min(n, i + cur_batch)
            try:
                batch_probs = run_batch(texts[i:end])
                probs_list.append(batch_probs)
                i = end
            except RuntimeError as e:
                err_msg = str(e).lower()
                # If OOM, try to reduce batch size and retry
                if "out of memory" in err_msg or "cuda out of memory" in err_msg:
                    print(f"⚠️ CUDA OOM with batch_size={cur_batch}. Trying smaller batch...")
                    if cur_batch == 1:
                        # Already at smallest batch on CUDA, fallback to CPU processing
                        print("⚠️ Smallest batch on CUDA failed — falling back to CPU inference.")
                        # move model to CPU and process remainder in tiny batches
                        self._move_model_to_cpu()
                        # process remaining items on CPU in batch=1
                        small_batch = 1
                        while i < n:
                            end2 = min(n, i + small_batch)
                            batch_probs = self._predict_on_cpu(texts[i:end2])
                            probs_list.append(batch_probs)
                            i = end2
                        break
                    # reduce batch size and retry the *same* window
                    cur_batch = max(1, cur_batch // 2)
                    continue
                else:
                    # Other runtime errors — re-raise
                    raise e

        if len(probs_list) == 0:
            return np.zeros((0, self.num_classes))
        return np.vstack(probs_list)

    def _predict_on_cpu(self, batch_texts):
        """Force CPU inference for given texts (batch_texts list)."""
        enc = self.tokenizer(
            batch_texts,
            add_special_tokens=True,
            max_length=self.max_length,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )
        enc = {k: v.to("cpu") for k, v in enc.items()}
        with torch.no_grad():
            out = self.model(**enc) if next(self.model.parameters()).device.type == "cpu" else self.model.to("cpu") and self.model(**enc)
            logits = out.logits / (self.temperature if isinstance(self.temperature, (float, int)) else float(self.temperature))
            probs = torch.softmax(logits, dim=1).cpu().numpy()
        return probs

    def _move_model_to_cpu(self):
        """Move model to CPU (useful when CUDA repeatedly OOMs)."""
        try:
            if self.device.type != "cpu":
                print("ℹ️ Moving model to CPU to avoid further OOMs.")
                self.model.to("cpu")
                self.device = torch.device("cpu")
                torch.cuda.empty_cache()
        except Exception as e:
            print(f"⚠️ Failed to move model to CPU ({e})")

    # -----------------------------------------------------------
    def predict_proba(self, texts, batch_size: int = 8):
        """Return softmax probabilities for one or more texts (batched, OOM-resilient)"""
        if not self.is_trained:
            raise ValueError("Model not trained or loaded.")

        # If model is on GPU but very small VRAM, user can pass smaller batch_size
        try:
            probs = self._batched_predict_proba(texts, batch_size=batch_size)
            return probs
        except RuntimeError as e:
            # If something else goes wrong, fallback to CPU single-item inference to be robust
            print(f"⚠️ Inference failed on device {self.device} (error: {e}). Falling back to CPU single-item inference.")
            self._move_model_to_cpu()
            if isinstance(texts, str):
                texts = [texts]
            all_probs = []
            for t in texts:
                all_probs.append(self._predict_on_cpu([t]))
            return np.vstack(all_probs)

    def predict_single(self, text):
        """Predict a single email with timing and confidence"""
        if not self.is_trained:
            raise ValueError("Model not trained or loaded.")
        start = time.time()
        probs = self.predict_proba([text], batch_size=1)[0]
        pred_idx = np.argmax(probs)
        pred_class = self.id2label[pred_idx]
        return {
            "prediction": pred_class,
            "confidence": float(probs[pred_idx]),
            "inference_time_ms": (time.time() - start) * 1000,
            "probabilities": probs.tolist(),
        }

    def predict(self, X):
        """Batch class prediction (accepts list-like of strings)"""
        probs = self.predict_proba(X)
        preds = np.argmax(probs, axis=1)
        return np.array([self.id2label[i] for i in preds])

    # -----------------------------------------------------------
    def save(self, path: Path):
        """Save model + metadata"""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        # Save model and tokenizer using transformers' save_pretrained
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        joblib.dump(
            {
                "classes": self.classes_,
                "label2id": self.label2id,
                "id2label": self.id2label,
                "max_length": self.max_length,
                "model_name": self.model_name,
                "training_time": self.training_time,
                "is_trained": self.is_trained,
                # ensure temperature is serializable
                "temperature": float(self.temperature) if not isinstance(self.temperature, (dict, list)) else self.temperature,
            },
            path / "metadata.pkl",
        )
        print(f"✅ Heavy model saved to: {path}")

    @classmethod
    def load(cls, path: Path):
        """Load model + metadata (keeps memory-smart behavior)"""
        path = Path(path)
        metadata = joblib.load(path / "metadata.pkl")

        model = cls(
            num_classes=len(metadata.get("classes", [])),
            max_length=metadata.get("max_length", 512),
            model_name=metadata.get("model_name", "microsoft/deberta-v3-base"),
        )
        # load tokenizer and model; model will be moved to model.device inside _create_model
        model.tokenizer = DebertaV2Tokenizer.from_pretrained(path)
        # Load model with low_cpu_mem_usage to reduce memory spikes
        try:
            model.model = DebertaV2ForSequenceClassification.from_pretrained(path, low_cpu_mem_usage=True)
        except Exception:
            model.model = DebertaV2ForSequenceClassification.from_pretrained(path)
        # Move to device
        model.model.to(model.device)
        model.model.eval()

        model.classes_ = metadata.get("classes", [])
        model.label2id = metadata.get("label2id", {})
        model.id2label = metadata.get("id2label", {})
        model.training_time = metadata.get("training_time", None)
        model.is_trained = metadata.get("is_trained", True)
        model.temperature = metadata.get("temperature", 1.0)
        print(f"✅ Heavy model loaded from: {path} (device={model.device})")
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
        learning_rate=1e-5,
    )

    model.save(Config.HEAVY_MODEL_PATH)
    print("\n✅ Heavy model training complete!")
    print(metrics)
