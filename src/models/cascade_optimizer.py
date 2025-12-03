"""
src/models/cascade_optimizer.py

Offline tool to calibrate probabilities and compute per-class thresholds for the
cascade router. Produces a JSON config consumed at runtime by CascadeClassifier.

Saves config to: data/models/cascade/green_ai_config.json (created automatically).

Usage (from project root):
    python src/models/cascade_optimizer.py

Requirements: torch, numpy, pandas
This script expects trained models saved at Config.GREEN_MODEL_PATH etc and a
validation CSV at Config.PROCESSED_DATA_DIR / "val.csv" with columns
'processed_text' and 'label'. Labels must be string class names present in
Config.CLASSES.
"""

import json
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, Any
import time

from src.config import Config
from src.models.green_model import GreenModel
from src.models.medium_model import MediumModel
from src.models.heavy_model import HeavyModel
import pandas as pd

EPS = 1e-12


# -----------------------------
# Temperature scaler (torch)
# -----------------------------
class TemperatureScaler(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.temperature = torch.nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, logits):
        return logits / self.temperature


def calibrate_temperature_from_logits(logits: np.ndarray, labels: np.ndarray, max_iter: int = 500) -> float:
    """Fit a single temperature parameter T using LBFGS on cross-entropy.

    logits: (N, C) numpy array (pseudo-logits allowed)
    labels: (N,) int
    Returns: scalar temperature (float)
    """
    device = torch.device("cpu")
    logits_t = torch.tensor(logits, dtype=torch.float32, device=device)
    labels_t = torch.tensor(labels, dtype=torch.long, device=device)

    model = TemperatureScaler().to(device)
    # LBFGS with PyTorch requires a closure
    optimizer = torch.optim.LBFGS([model.temperature], lr=0.01, max_iter=max_iter)
    loss_fn = torch.nn.CrossEntropyLoss()

    def closure():
        optimizer.zero_grad()
        scaled = model(logits_t)
        loss = loss_fn(scaled, labels_t)
        loss.backward()
        return loss

    # run optimizer
    optimizer.step(closure)
    return float(model.temperature.detach().cpu().numpy())


# -----------------------------
# Utilities: convert probs -> "pseudo logits"
# -----------------------------

def probs_to_logits(probs: np.ndarray) -> np.ndarray:
    """Safe log transform to create pseudo-logits from probabilities."""
    probs = np.clip(probs, EPS, 1.0 - EPS)
    return np.log(probs)


# -----------------------------
# Per-class threshold computation
# -----------------------------

def compute_per_class_thresholds(
    probs: np.ndarray,
    true_labels: np.ndarray,
    acc_req_per_class: Dict[int, float],
    threshold_grid: np.ndarray = np.arange(0.50, 0.96, 0.01),
) -> Dict[int, float]:
    """
    Returns per-class threshold mapping {class_idx: threshold}.
    probs: (N, C) calibrated probabilities (sum to 1)
    true_labels: (N,) ints
    """
    pred_classes = np.argmax(probs, axis=1)
    conf_per_email = probs[np.arange(len(probs)), pred_classes]

    thresholds = {}
    num_classes = probs.shape[1]

    for cls in range(num_classes):
        mask = pred_classes == cls
        if mask.sum() == 0:
            thresholds[cls] = 0.5
            continue

        cls_conf = conf_per_email[mask]
        cls_true = true_labels[mask]
        cls_pred = pred_classes[mask]

        best_T = 0.50
        for T in threshold_grid:
            sel = cls_conf >= T
            if sel.sum() == 0:
                continue
            acc = (cls_true[sel] == cls_pred[sel]).mean()
            if acc >= acc_req_per_class.get(cls, 0.9):
                best_T = float(T)
                break
        thresholds[cls] = float(best_T)

    return thresholds


# -----------------------------
# Routing evaluation (on val set)
# -----------------------------

def evaluate_routing(lr_probs, med_probs, heavy_probs, true_labels, T1, T2):
    final_preds = []
    model_used = []

    lr_pred = np.argmax(lr_probs, axis=1)
    lr_conf = lr_probs[np.arange(len(lr_probs)), lr_pred]

    med_pred = np.argmax(med_probs, axis=1)
    med_conf = med_probs[np.arange(len(med_probs)), med_pred]

    heavy_pred = np.argmax(heavy_probs, axis=1)

    for i in range(len(true_labels)):
        if lr_conf[i] >= T1[lr_pred[i]]:
            final_preds.append(lr_pred[i])
            model_used.append("LR")
            continue
        if med_conf[i] >= T2[med_pred[i]]:
            final_preds.append(med_pred[i])
            model_used.append("Medium")
            continue
        final_preds.append(heavy_pred[i])
        model_used.append("Heavy")

    final_preds = np.array(final_preds)
    model_used = np.array(model_used)
    acc = float((final_preds == true_labels).mean())

    return {
        "accuracy": acc,
        "lr_count": int((model_used == "LR").sum()),
        "med_count": int((model_used == "Medium").sum()),
        "heavy_count": int((model_used == "Heavy").sum()),
    }


# -----------------------------
# High level fit function
# -----------------------------

def fit_cascade_config(
    lr_probs_val: np.ndarray,
    med_probs_val: np.ndarray,
    heavy_probs_val: np.ndarray,
    true_labels_val: np.ndarray,
    acc_req_lr: Dict[int, float] = None,
    acc_req_med: Dict[int, float] = None,
):
    num_classes = lr_probs_val.shape[1]

    if acc_req_lr is None:
        acc_req_lr = {c: 0.88 for c in range(num_classes)}
    if acc_req_med is None:
        acc_req_med = {c: 0.95 for c in range(num_classes)}

    # Convert probs -> pseudo-logits
    lr_logits = probs_to_logits(lr_probs_val)
    med_logits = probs_to_logits(med_probs_val)
    heavy_logits = probs_to_logits(heavy_probs_val)

    # Fit temperatures
    print("🔢 Calibrating temperatures (this may take a moment)...")
    t0 = time.time()
    T_lr = calibrate_temperature_from_logits(lr_logits, true_labels_val)
    T_med = calibrate_temperature_from_logits(med_logits, true_labels_val)
    t_elapsed = time.time() - t0
    print(f"✅ Temperatures fitted in {t_elapsed:.1f}s — T_lr={T_lr:.3f}, T_med={T_med:.3f}")

    # Apply temperatures to logits => calibrated probs
    lr_probs = F.softmax(torch.tensor(lr_logits) / T_lr, dim=1).numpy()
    med_probs = F.softmax(torch.tensor(med_logits) / T_med, dim=1).numpy()
    heavy_probs = F.softmax(torch.tensor(heavy_logits), dim=1).numpy()

    # Compute per-class thresholds
    T1 = compute_per_class_thresholds(lr_probs, true_labels_val, acc_req_lr)
    T2 = compute_per_class_thresholds(med_probs, true_labels_val, acc_req_med)

    # Evaluate routing
    stats = evaluate_routing(lr_probs, med_probs, heavy_probs, true_labels_val, T1, T2)

    config = {
        "temperature_lr": float(T_lr),
        "temperature_med": float(T_med),
        "thresholds_lr": {str(k): float(v) for k, v in T1.items()},
        "thresholds_med": {str(k): float(v) for k, v in T2.items()},
        "num_classes": int(num_classes),
        "validation_stats": stats,
    }
    return config


# -----------------------------
# CLI / Main
# -----------------------------

def main():
    print("=" * 70)
    print("🔍 Fitting Cascade Thresholds & Calibration")
    print("=" * 70)

    # Load validation dataset
    val_path = Config.PROCESSED_DATA_DIR / "val.csv"
    if not val_path.exists():
        raise SystemExit(f"❌ Validation CSV not found: {val_path}")

    val_df = pd.read_csv(val_path)
    print(f"✅ Loaded {len(val_df)} validation samples")

    X_val = val_df["processed_text"].tolist()
    y_val = val_df["label"].tolist()

    # Map labels to integers consistent with Config.CLASSES
    classes = list(Config.CLASSES)
    label_to_idx = {lbl: i for i, lbl in enumerate(classes)}
    y_val_idx = np.array([label_to_idx[s] for s in y_val])

    # Load models
    print("📦 Loading models...")
    green = GreenModel.load(Config.GREEN_MODEL_PATH)
    med = MediumModel.load(Config.MEDIUM_MODEL_PATH)
    heavy = HeavyModel.load(Config.HEAVY_MODEL_PATH)

    # Get probabilities on validation set
    print("🔢 Getting Green (LR) probabilities on val set...")
    lr_probs = np.vstack([p for p in green.predict_proba(X_val)])

    print("🔢 Getting Medium probabilities on val set...")
    med_probs = np.vstack([p for p in med.predict_proba(X_val)])

    print("🔢 Getting Heavy probabilities on val set...")
    heavy_probs = np.vstack([p for p in heavy.predict_proba(X_val)])

    # Fit config
    config = fit_cascade_config(lr_probs, med_probs, heavy_probs, y_val_idx,
                                acc_req_lr={i: 0.88 for i in range(len(classes))},
                                acc_req_med={i: 0.95 for i in range(len(classes))})

    # Save config
    out_dir = Config.MODELS_DIR / "cascade"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "green_ai_config.json"
    with open(out_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"✅ Saved cascade config to: {out_path}")
    print("Validation stats:", config.get("validation_stats"))


if __name__ == "__main__":
    main()
