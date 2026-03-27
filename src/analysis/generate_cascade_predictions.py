import sys
from pathlib import Path

# -------------------------------------------------
# Ensure project root is on PYTHONPATH
# -------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

import pandas as pd
import requests
from src.config import Config

API_URL = f"http://localhost:{Config.API_PORT}/classify"

# Load test set (or val.csv for paper)
df = pd.read_csv("data/processed/test.csv")

rows = []

for _, row in df.iterrows():
    text = row["processed_text"]
    true_label = row["label"]

    r = requests.post(API_URL, json={"text": text})
    result = r.json()

    rows.append({
        "true_label": true_label,
        "predicted_label": result["prediction"],
        "model_used": result["model_used"],
        "confidence": result["confidence"],
        "energy_kwh": result.get("energy_kwh", 0.0),
        "co2_grams": result.get("co2_grams", 0.0),
    })

out = pd.DataFrame(rows)

Path("data").mkdir(exist_ok=True)
out.to_csv("data/cascade_predictions.csv", index=False)

print("✅ data/cascade_predictions.csv generated")
print(out.head())
