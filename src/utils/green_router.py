import json
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from typing import Dict, Any, Optional

EPS = 1e-12

# ============================================================
# 1. Temperature Scaling (Calibration)
# ============================================================

class TemperatureScaler(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.temperature = torch.nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, logits):
        return logits / self.temperature


def calibrate_temperature_from_logits(logits: np.ndarray, labels: np.ndarray, max_iter=500) -> float:
    """Fit a single temperature scalar T using LBFGS on cross-entropy.

    This runs on CPU (safe & reproducible) and returns a float T.
    """
    device = torch.device("cpu")
    logits_t = torch.tensor(logits, dtype=torch.float32, device=device)
    labels_t = torch.tensor(labels, dtype=torch.long, device=device)

    model = TemperatureScaler().to(device)
    optimizer = torch.optim.LBFGS([model.temperature], lr=0.01, max_iter=max_iter)
    loss_fn = torch.nn.CrossEntropyLoss()

    def closure():
        optimizer.zero_grad()
        loss = loss_fn(model(logits_t), labels_t)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(model.temperature.detach().cpu().numpy())

# ============================================================
# 2. Utils: Convert probabilities → pseudo-logits
# ============================================================

def probs_to_logits(probs: np.ndarray):
    probs = np.clip(probs, EPS, 1.0 - EPS)
    return np.log(probs)

# ============================================================
# 3. Compute Per-Class Confidence Thresholds
# ============================================================

def compute_per_class_thresholds(
    probs: np.ndarray,
    true_labels: np.ndarray,
    acc_req_per_class: Dict[int, float],
    threshold_grid: np.ndarray = np.arange(0.50, 0.96, 0.01),
) -> Dict[int, float]:

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
            selected = cls_conf >= T
            if selected.sum() == 0:
                continue

            acc = (cls_true[selected] == cls_pred[selected]).mean()

            if acc >= acc_req_per_class.get(cls, 0.9):
                best_T = float(T)
                break

        thresholds[cls] = best_T

    return thresholds

# ============================================================
# 4. Validation Routing Evaluation
# ============================================================

def evaluate_routing(lr_probs, med_probs, heavy_probs, true_labels, T1, T2):
    final_preds = []
    model_used = []

    lr_pred = np.argmax(lr_probs, axis=1)
    lr_conf = lr_probs[np.arange(len(lr_probs)), lr_pred]

    med_pred = np.argmax(med_probs, axis=1)
    med_conf = med_probs[np.arange(len(med_probs)), med_pred]

    heavy_pred = np.argmax(heavy_probs, axis=1)

    for i in range(len(true_labels)):
        if lr_conf[i] >= T1[int(lr_pred[i])]:
            final_preds.append(int(lr_pred[i]))
            model_used.append("LR")
            continue

        if med_conf[i] >= T2[int(med_pred[i])]:
            final_preds.append(int(med_pred[i]))
            model_used.append("Medium")
            continue

        final_preds.append(int(heavy_pred[i]))
        model_used.append("Heavy")

    final_preds = np.array(final_preds)
    acc = float(accuracy_score(true_labels, final_preds))

    return {
        "accuracy": acc,
        "lr_count": int((np.array(model_used) == "LR").sum()),
        "med_count": int((np.array(model_used) == "Medium").sum()),
        "heavy_count": int((np.array(model_used) == "Heavy").sum()),
    }

# ============================================================
# 5. High-level Training-side Function (Used Offline)
#
# NOTE: Backwards-compatible signature — accepts several common
#       keyword variants so scripts won't break.
# ============================================================

def fit_green_router(
    lr_probs_val: Optional[np.ndarray] = None,
    med_probs_val: Optional[np.ndarray] = None,
    heavy_probs_val: Optional[np.ndarray] = None,
    true_labels_val: Optional[np.ndarray] = None,
    acc_req_lr: Optional[Dict[int, float]] = None,
    acc_req_med: Optional[Dict[int, float]] = None,
    threshold_grid=np.arange(0.50, 0.96, 0.01),
    # alias params (accepted for backwards compatibility)
    lr_probs: Optional[np.ndarray] = None,
    med_probs: Optional[np.ndarray] = None,
    heavy_probs: Optional[np.ndarray] = None,
    y_true: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
) -> Dict[str, Any]:

    """Fit calibration (temperature) and per-class thresholds on validation set.

    This function is tolerant about input argument names — your script can
    call it using either positional args or older keyword names like
    `lr_probs=` / `y_true=`.

    Returns a dict suitable for saving as JSON and loading in production.
    """

    # -------------------------
    # Resolve aliases (backwards compatibility)
    # -------------------------
    if lr_probs_val is None and lr_probs is not None:
        lr_probs_val = lr_probs
    if med_probs_val is None and med_probs is not None:
        med_probs_val = med_probs
    if heavy_probs_val is None and heavy_probs is not None:
        heavy_probs_val = heavy_probs

    if true_labels_val is None:
        if y_true is not None:
            true_labels_val = y_true
        elif y_val is not None:
            true_labels_val = y_val

    # Basic validation
    if lr_probs_val is None or med_probs_val is None or heavy_probs_val is None or true_labels_val is None:
        raise ValueError("Missing required inputs. Provide lr_probs_val, med_probs_val, heavy_probs_val, and true_labels_val (or their aliases).")

    # Ensure numpy arrays
    lr_probs_val = np.asarray(lr_probs_val)
    med_probs_val = np.asarray(med_probs_val)
    heavy_probs_val = np.asarray(heavy_probs_val)
    true_labels_val = np.asarray(true_labels_val)

    num_classes = lr_probs_val.shape[1]

    if acc_req_lr is None:
        acc_req_lr = {c: 0.88 for c in range(num_classes)}
    if acc_req_med is None:
        acc_req_med = {c: 0.95 for c in range(num_classes)}

    # Convert probs → pseudo logits
    lr_logits = probs_to_logits(lr_probs_val)
    med_logits = probs_to_logits(med_probs_val)
    heavy_logits = probs_to_logits(heavy_probs_val)

    # Temperature scaling (learn T on CPU)
    T_lr = calibrate_temperature_from_logits(lr_logits, true_labels_val)
    T_med = calibrate_temperature_from_logits(med_logits, true_labels_val)

    # Calibrated probabilities
    lr_probs_cal = F.softmax(torch.tensor(lr_logits) / T_lr, dim=1).numpy()
    med_probs_cal = F.softmax(torch.tensor(med_logits) / T_med, dim=1).numpy()
    heavy_probs_cal = F.softmax(torch.tensor(heavy_logits), dim=1).numpy()

    # Compute per-class thresholds (only for LR & Medium)
    T1 = compute_per_class_thresholds(lr_probs_cal, true_labels_val, acc_req_lr, threshold_grid)
    T2 = compute_per_class_thresholds(med_probs_cal, true_labels_val, acc_req_med, threshold_grid)

    # Validation stats
    stats = evaluate_routing(lr_probs_cal, med_probs_cal, heavy_probs_cal, true_labels_val, T1, T2)

    config = {
        "temperature_lr": float(T_lr),
        "temperature_med": float(T_med),
        "thresholds_lr": {str(c): float(v) for c, v in T1.items()},
        "thresholds_med": {str(c): float(v) for c, v in T2.items()},
        "num_classes": int(num_classes),
        "validation_stats": stats,
    }

    return config

# ============================================================
# 6. Runtime Router (Production)
# ============================================================

class GreenRouter:
    """
    Upgraded Cascade-style runtime router.

    Usage:
        router = GreenRouter.from_config_path(path, green_model, medium_model, heavy_model)
        pred_idx, model_used, details = router.predict_single(text)
    """

    def __init__(self, config: Dict[str, Any], lr_model, med_model, heavy_model):
        self.num_classes = config["num_classes"]
        self.T_lr = float(config["temperature_lr"])
        self.T_med = float(config["temperature_med"])
        self.T1 = {int(k): float(v) for k, v in config["thresholds_lr"].items()}
        self.T2 = {int(k): float(v) for k, v in config["thresholds_med"].items()}

        self.lr_model = lr_model
        self.med_model = med_model
        self.heavy_model = heavy_model

    @classmethod
    def from_config_path(cls, path, lr_model, med_model, heavy_model):
        with open(path, "r") as f:
            cfg = json.load(f)
        return cls(cfg, lr_model, med_model, heavy_model)

    def _probs_from_logits(self, logits, temperature=None):
        logits_t = torch.tensor(logits, dtype=torch.float32)
        if temperature is not None:
            logits_t = logits_t / temperature
        return F.softmax(logits_t, dim=-1).detach().cpu().numpy()

    def predict_single(self, text: str):
        """Return (pred_idx:int, model_used:str, details:dict)

        Models should expose `predict_proba([text])` returning shape (1, C) probabilities
        or logits. This router safely detects and handles both.
        """

        # ------------------------
        # Tier 1 — GREEN (light)
        # ------------------------
        lr_out = np.asarray(self.lr_model.predict_proba([text]))
        if lr_out.ndim == 2:
            lr_logits = probs_to_logits(lr_out)
        else:
            lr_logits = np.asarray(lr_out)

        lr_p = self._probs_from_logits(lr_logits, self.T_lr)[0]
        lr_pred = int(np.argmax(lr_p))
        lr_conf = float(lr_p[lr_pred])

        if lr_conf >= self.T1.get(lr_pred, 0.5):
            return lr_pred, "green", {"probs": lr_p.tolist()}

        # ------------------------
        # Tier 2 — MEDIUM
        # ------------------------
        med_out = np.asarray(self.med_model.predict_proba([text]))
        if med_out.ndim == 2:
            med_logits = probs_to_logits(med_out)
        else:
            med_logits = np.asarray(med_out)

        med_p = self._probs_from_logits(med_logits, self.T_med)[0]
        med_pred = int(np.argmax(med_p))
        med_conf = float(med_p[med_pred])

        if med_conf >= self.T2.get(med_pred, 0.5):
            return med_pred, "medium", {"probs": med_p.tolist()}

        # ------------------------
        # Tier 3 — HEAVY (fallback)
        # ------------------------
        heavy_out = np.asarray(self.heavy_model.predict_proba([text]))
        if heavy_out.ndim == 2:
            heavy_p = heavy_out[0]
        else:
            heavy_p = F.softmax(torch.tensor(heavy_out), dim=-1).detach().cpu().numpy()[0]

        heavy_pred = int(np.argmax(heavy_p))
        return heavy_pred, "heavy", {"probs": heavy_p.tolist()}
