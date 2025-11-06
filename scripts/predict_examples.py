#!/usr/bin/env python3
"""
Run predictions on sample email texts using a trained model.
✅ Works with Green, Medium, or Heavy model versions.
"""

import sys
from pathlib import Path

# Add project root to path
repo_root = Path(__file__).resolve().parent.parent
sys.path.append(str(repo_root))

from src.config import Config
from src.models.heavy_model import HeavyModel  # change to GreenModel or HeavyModel if needed


# ------------------------------
# Load trained model
# ------------------------------
print("🔄 Loading trained model...")
model = HeavyModel.load(Config.HEAVY_MODEL_PATH)
print("✅ Model loaded successfully!\n")

# ------------------------------
# Example emails
# ------------------------------
examples = [
    # Example 1: Work-related
    "Team, please review the updated project roadmap before tomorrow’s meeting. We’ll finalize milestones for the next sprint.",

    # Example 2: Support-related
    "Hi, my account has been locked after multiple login attempts. Can you please reset my password or help me regain access?",

    # Example 3: Spam/Promotional
    "🎉 Congratulations! You’ve won a $500 Amazon gift card. Click the link to claim your reward before it expires!",

    # Example 4: Obvious spam
    "🎉Congratulations! Lalit you won 2cr rupees in a contest grab it before the reward expires"
]

# ------------------------------
# Run predictions
# ------------------------------
print("📨 Predicting categories...\n")

for text in examples:
    result = model.predict_single(text)
    print(f"📧 Email: {text}")
    print(f"→ Predicted Category: {result['prediction']}  (Confidence: {result['confidence']:.2f})\n")
