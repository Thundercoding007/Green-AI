# scripts/fit_green_router.py

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import json
import numpy as np
import pandas as pd

from src.utils.green_router import fit_green_router
from src.config import Config
from src.models.green_model import GreenModel
from src.models.medium_model import MediumModel
from src.models.heavy_model import HeavyModel


def main():
    print("\n===============================================")
    print("🚀 Running Green Router Fitter (calibration)")
    print("===============================================\n")

    # --------------------------------------------------------
    # 1️⃣ Load Models
    # --------------------------------------------------------
    print("📦 Loading Green Model...")
    green = GreenModel.load(Config.GREEN_MODEL_PATH)

    print("📦 Loading Medium Model...")
    medium = MediumModel.load(Config.MEDIUM_MODEL_PATH)

    print("📦 Loading Heavy Model (forced CPU to avoid GPU OOM)...")
    heavy = HeavyModel.load(Config.HEAVY_MODEL_PATH)

    # 🔥 Force heavy model to CPU for safety
    import torch
    heavy.device = torch.device("cpu")
    heavy.model.to("cpu")
    print("   ✔ Heavy model moved to CPU\n")

    # --------------------------------------------------------
    # 2️⃣ Load Validation Set
    # --------------------------------------------------------
    print("📄 Loading validation dataset...")
    val_df = pd.read_csv(Config.PROCESSED_DATA_DIR / "val.csv")
    X_val = val_df["processed_text"].tolist()
    y_val = val_df["label"].tolist()

    # Convert string labels → numeric id
    classes = list(Config.CLASSES)
    label_to_idx = {lbl: i for i, lbl in enumerate(classes)}
    y_val_idx = np.array([label_to_idx[s] for s in y_val])

    # --------------------------------------------------------
    # 3️⃣ Compute Model Probabilities
    # --------------------------------------------------------
    print("\n🔢 Getting LR (green) probabilities on val set...")
    lr_probs = green.predict_proba(X_val)

    print("🔢 Getting Medium probabilities on val set...")
    med_probs = medium.predict_proba(X_val)

    print("🔢 Getting Heavy probabilities on val set...")
    heavy_probs = heavy.predict_proba(X_val)   # heavy is CPU-safe now

    # Stack into N×C arrays
    lr_probs = np.vstack(lr_probs)
    med_probs = np.vstack(med_probs)
    heavy_probs = np.vstack(heavy_probs)

    # --------------------------------------------------------
    # 4️⃣ Fit Router
    # --------------------------------------------------------
    print("\n🧠 Fitting Smart Cascade Router (per-class thresholds)...")

    config = fit_green_router(
        lr_probs=lr_probs,
        med_probs=med_probs,
        heavy_probs=heavy_probs,
        y_true=y_val_idx,
        acc_req_lr={i: 0.88 for i in range(len(classes))},
        acc_req_med={i: 0.95 for i in range(len(classes))}
    )

    # --------------------------------------------------------
    # 5️⃣ Save config JSON
    # --------------------------------------------------------
    out_dir = Config.MODELS_DIR / "cascade"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / "green_ai_config.json"
    with open(out_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"\n✅ Saved router config → {out_path}")
    print("📊 Validation stats:\n", config.get("validation_stats"))


if __name__ == "__main__":
    main()
