# scripts/debug_router_checks.py
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np, pandas as pd
from pathlib import Path
from src.config import Config
from src.models.green_model import GreenModel
from src.models.medium_model import MediumModel
from src.models.heavy_model import HeavyModel

green = GreenModel.load(Config.GREEN_MODEL_PATH)
medium = MediumModel.load(Config.MEDIUM_MODEL_PATH)
heavy = HeavyModel.load(Config.HEAVY_MODEL_PATH)

print("Config.CLASSES:", Config.CLASSES)
print("Green classes attr:", getattr(green, "classes_", None))
print("Medium classes attr:", getattr(medium, "classes_", None))
print("Heavy id2label:", getattr(heavy, "id2label", None))

val = pd.read_csv(Config.PROCESSED_DATA_DIR / "val.csv")
print("val.csv unique labels:", sorted(val['label'].unique())[:20])
print("first 10 val labels:", val['label'].head(10).tolist())

X_val = val['processed_text'].tolist()

# Get probs for first 5 examples
lr_p = np.vstack([p for p in green.predict_proba(X_val[:5])])
med_p = np.vstack([p for p in medium.predict_proba(X_val[:5])])
heavy_p = np.vstack([p for p in heavy.predict_proba(X_val[:5])])

print("shapes (first5) lr,med,heavy:", lr_p.shape, med_p.shape, heavy_p.shape)
print("lr probs sums:", lr_p.sum(axis=1))
print("med probs sums:", med_p.sum(axis=1))
print("heavy probs sums:", heavy_p.sum(axis=1))

# quick sample predictions (strings if available)
def idx2label_from_model(m, idx):
    if hasattr(m, "id2label") and m.id2label:
        return m.id2label.get(int(idx), None)
    if hasattr(m, "classes_") and m.classes_ is not None:
        return m.classes_[int(idx)]
    return None

for i in range(5):
    lr_idx = int(lr_p[i].argmax()); med_idx = int(med_p[i].argmax()); hv_idx = int(heavy_p[i].argmax())
    print(f"[{i}] lr_idx:{lr_idx} => {idx2label_from_model(green, lr_idx)}, med_idx:{med_idx} => {idx2label_from_model(medium, med_idx)}, heavy_idx:{hv_idx} => {idx2label_from_model(heavy, hv_idx)}")
