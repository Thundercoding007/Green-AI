# File: src/analysis/cascade_evaluation_plots.py
# Research-grade evaluation plots for GreenAI Cascade System

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score

# -----------------------------
# CONFIG
# -----------------------------
CLASSES = ["spam", "work", "support"]
OUTPUT_DIR = "src/analysis/outputs"
CSV_PATH = "data/cascade_predictions.csv"  # <-- change if needed

# Ensure output dir exists
import os
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------
# LOAD DATA
# -----------------------------
df = pd.read_csv(CSV_PATH)

y_true = df["true_label"]
y_pred = df["predicted_label"]

# -----------------------------
# 1. CONFUSION MATRIX
# -----------------------------
cm = confusion_matrix(y_true, y_pred, labels=CLASSES)

plt.figure(figsize=(7, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=CLASSES,
    yticklabels=CLASSES,
)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix — Final Cascaded System")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/confusion_matrix.png", dpi=300)
plt.close()

# -----------------------------
# 2. ERROR DISTRIBUTION PER CLASS
# -----------------------------
errors = df[df["true_label"] != df["predicted_label"]]
error_counts = errors["true_label"].value_counts().reindex(CLASSES, fill_value=0)

plt.figure(figsize=(7, 5))
sns.barplot(x=error_counts.index, y=error_counts.values, palette="Reds")
plt.ylabel("Number of Misclassifications")
plt.xlabel("Class")
plt.title("Error Distribution per Class")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/error_distribution.png", dpi=300)
plt.close()

# -----------------------------
# 3. ACTUAL vs PREDICTED DISTRIBUTION
# -----------------------------
actual_counts = y_true.value_counts().reindex(CLASSES, fill_value=0)
pred_counts = y_pred.value_counts().reindex(CLASSES, fill_value=0)

dist_df = pd.DataFrame({
    "Class": CLASSES,
    "Actual": actual_counts.values,
    "Predicted": pred_counts.values,
})

x = np.arange(len(CLASSES))
width = 0.35

plt.figure(figsize=(8, 5))
plt.bar(x - width/2, dist_df["Actual"], width, label="Actual")
plt.bar(x + width/2, dist_df["Predicted"], width, label="Predicted")

plt.xticks(x, CLASSES)
plt.ylabel("Count")
plt.title("Actual vs Predicted Class Distribution")
plt.legend()
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/actual_vs_predicted.png", dpi=300)
plt.close()

# -----------------------------
# 4. ESCALATION RATE vs ACCURACY
# -----------------------------
# Escalation = fraction routed to HEAVY
df["is_heavy"] = df["model_used"] == "heavy"

thresholds = np.linspace(0.0, 1.0, 11)
accs = []
escalations = []

for t in thresholds:
    subset = df[df["confidence"] >= t]
    if len(subset) == 0:
        continue

    acc = accuracy_score(subset["true_label"], subset["predicted_label"])
    escalation_rate = subset["is_heavy"].mean()

    accs.append(acc)
    escalations.append(escalation_rate * 100)

plt.figure(figsize=(7, 5))
plt.plot(escalations, accs, marker="o")
plt.xlabel("% Emails Escalated to Heavy Model")
plt.ylabel("Accuracy")
plt.title("Escalation Rate vs Accuracy")
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/escalation_vs_accuracy.png", dpi=300)
plt.close()

# -----------------------------
# 5. ENERGY / CO₂ vs ACCURACY
# -----------------------------
summary = []

for model in ["green", "medium", "heavy"]:
    subset = df[df["model_used"] == model]
    if len(subset) == 0:
        continue

    acc = accuracy_score(subset["true_label"], subset["predicted_label"])
    energy = subset["energy_kwh"].mean()

    summary.append((model, energy, acc))

# Cascaded system
summary.append((
    "cascade",
    df["energy_kwh"].mean(),
    accuracy_score(y_true, y_pred),
))

summary_df = pd.DataFrame(summary, columns=["Model", "Energy_kWh", "Accuracy"])

plt.figure(figsize=(7, 5))
sns.scatterplot(
    data=summary_df,
    x="Energy_kWh",
    y="Accuracy",
    hue="Model",
    s=120
)
plt.xlabel("Energy per Inference (kWh)")
plt.ylabel("Accuracy")
plt.title("Energy vs Accuracy Tradeoff")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/energy_vs_accuracy.png", dpi=300)
plt.close()

print("✅ All research plots generated successfully!")
print(f"📁 Saved to: {OUTPUT_DIR}")
