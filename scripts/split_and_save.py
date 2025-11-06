#!/usr/bin/env python3
"""
Clean, validate, balance, and split a single CSV dataset into train/val/test sets.
✅ VS Code compatible — uses project Config and utilities from src.
"""

import re
import json
from pathlib import Path
import pandas as pd

# Import Config and preprocessing utilities
from src.config import Config
from src.utils.preprocessing import (
    split_dataset,
    EmailPreprocessor,
    create_balanced_dataset,
)


# ---------------------------------------------------------------------
# 1️⃣ Locate raw dataset
# ---------------------------------------------------------------------
raw_dir = Config.RAW_DATA_DIR
csv_files = list(raw_dir.glob("*.csv"))

if not csv_files:
    raise SystemExit(f"❌ No CSV files found in {raw_dir}. Please place your dataset there.")

CSV_PATH = csv_files[0]
print(f"\n📄 Processing file: {CSV_PATH.name}")

# ---------------------------------------------------------------------
# 2️⃣ Load dataset
# ---------------------------------------------------------------------
df = pd.read_csv(CSV_PATH)
print("\n📊 Columns:", df.columns.tolist())
print("\n📋 Sample rows:")
print(df.head(5).to_string())

# ---------------------------------------------------------------------
# 3️⃣ Identify text column
# ---------------------------------------------------------------------
text_column = None
for possible_name in ["text", "clean_text", "email_text", "content", "message"]:
    if possible_name in df.columns:
        text_column = possible_name
        break

if not text_column:
    raise SystemExit("❌ No text column found. Expected one of: text, clean_text, email_text, content, message")

# ---------------------------------------------------------------------
# 4️⃣ Basic fallback cleaner
# ---------------------------------------------------------------------
def clean_email_text(text):
    """Simple text cleaner (used if EmailPreprocessor unavailable)."""
    if isinstance(text, str):
        text = re.sub(
            r"(from|to|subject|date|received|message-id|mime-version|content-type|x-|message id|return-path):.*?\n",
            "",
            text,
            flags=re.I | re.M,
        )
        text = re.sub(r"\S+@\S+\.\S+", "[EMAIL]", text)
        text = re.sub(
            r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+",
            "[URL]",
            text,
        )
        text = re.sub(r"[^\w\s.,!?;:\-']", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text
    return ""

# ---------------------------------------------------------------------
# 5️⃣ Clean and normalize text
# ---------------------------------------------------------------------
print(f"\n🧹 Cleaning email text using column '{text_column}'...")

try:
    preprocessor = EmailPreprocessor()
    df["processed_text"] = df[text_column].apply(lambda x: preprocessor.process_email(str(x), anonymize=True))
    print("✓ Cleaned and anonymized using EmailPreprocessor.")
except Exception as e:
    print(f"⚠️ EmailPreprocessor unavailable ({e}), falling back to basic cleaner.")
    df["processed_text"] = df[text_column].apply(clean_email_text)

# ---------------------------------------------------------------------
# 6️⃣ Validate label column
# ---------------------------------------------------------------------
if "label" not in df.columns:
    raise SystemExit('❌ No "label" column found. Please ensure your dataset includes labels.')

# ---------------------------------------------------------------------
# 7️⃣ Filter out too-short or too-long emails
# ---------------------------------------------------------------------
before = len(df)
df["processed_text"] = df["processed_text"].astype(str)
df = df[df["processed_text"].str.len() >= 10]
df = df[df["processed_text"].str.len() <= 5000]
df = df.reset_index(drop=True)
after = len(df)

print(f"\n🧾 Rows before: {before}, after filtering (10–5000 chars): {after}")

if after == 0:
    raise SystemExit("❌ No data left after cleaning. Check cleaning logic or thresholds.")

# ---------------------------------------------------------------------
# 8️⃣ Check class distribution
# ---------------------------------------------------------------------
print("\n📈 Class distribution:")
print(df["label"].value_counts())

# ---------------------------------------------------------------------
# 9️⃣ Create balanced dataset (optional)
# ---------------------------------------------------------------------
print("\n⚖️ Creating balanced dataset...")
try:
    balanced_df = create_balanced_dataset(df, samples_per_class=5000)
    print(f"✅ Balanced dataset created with {len(balanced_df)} samples.")
except Exception as e:
    print(f"⚠️ Could not balance dataset: {e}")
    print("➡️ Proceeding with original dataset.")
    balanced_df = df

# ---------------------------------------------------------------------
# 🔟 Split into train/val/test sets
# ---------------------------------------------------------------------
Config.ensure_directories()

try:
    train_df, val_df, test_df = split_dataset(
        balanced_df,
        test_size=Config.TEST_SIZE,
        val_size=Config.VAL_SIZE,
        random_state=Config.RANDOM_STATE,
    )
except ValueError as e:
    print(f"\n⚠️ Stratified split failed: {e}")
    print("➡️ Falling back to random split.")
    from sklearn.model_selection import train_test_split

    train_df, temp_df = train_test_split(
        balanced_df,
        test_size=Config.TEST_SIZE + Config.VAL_SIZE,
        random_state=Config.RANDOM_STATE,
        shuffle=True,
    )
    val_rel_size = Config.VAL_SIZE / (Config.TEST_SIZE + Config.VAL_SIZE)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=1 - val_rel_size,
        random_state=Config.RANDOM_STATE,
        shuffle=True,
    )

# ---------------------------------------------------------------------
# 11️⃣ Save processed splits
# ---------------------------------------------------------------------
train_path = Config.PROCESSED_DATA_DIR / "train.csv"
val_path = Config.PROCESSED_DATA_DIR / "val.csv"
test_path = Config.PROCESSED_DATA_DIR / "test.csv"

train_df.to_csv(train_path, index=False)
val_df.to_csv(val_path, index=False)
test_df.to_csv(test_path, index=False)

print("\n✅ Saved processed splits:")
print("  -", train_path)
print("  -", val_path)
print("  -", test_path)

# ---------------------------------------------------------------------
# 12️⃣ Save dataset statistics
# ---------------------------------------------------------------------
print("\n💾 Saving dataset statistics...")
stats = {
    "total_samples": len(balanced_df),
    "train_samples": len(train_df),
    "val_samples": len(val_df),
    "test_samples": len(test_df),
    "num_classes": len(balanced_df["label"].unique()),
    "classes": balanced_df["label"].unique().tolist(),
    "avg_text_length": balanced_df["processed_text"].str.len().mean(),
    "min_text_length": balanced_df["processed_text"].str.len().min(),
    "max_text_length": balanced_df["processed_text"].str.len().max(),
}

stats_path = Config.PROCESSED_DATA_DIR / "dataset_stats.json"
with open(stats_path, "w") as f:
    json.dump(stats, f, indent=2, default=str)
print(f"✅ Statistics saved to: {stats_path}")

# ---------------------------------------------------------------------
# 13️⃣ Final summary
# ---------------------------------------------------------------------
print("\n📊 Final split summary:")
print(f"Train: {len(train_df)} samples")
print(f"Val:   {len(val_df)} samples")
print(f"Test:  {len(test_df)} samples")

print("\n🎉 Done! Dataset cleaned, balanced, and split successfully.")
