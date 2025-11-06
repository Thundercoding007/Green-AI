# File: scripts/optimize_thresholds.py
# Optimize Cascade Thresholds using Validation Set (Aligned Version)

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import json

# Add parent project root
sys.path.append(str(Path(__file__).parent.parent))

from src.config import Config
from src.models.cascade import CascadeClassifier


# ---------------------------------------------------------------------
def plot_threshold_heatmap(results_df, output_path: Path):
    """Visualize grid of accuracy and energy savings."""
    print("\n📊 Generating threshold heatmaps...")
    output_path.mkdir(parents=True, exist_ok=True)

    acc_pivot = results_df.pivot(
        index="medium_threshold", columns="green_threshold", values="accuracy"
    )
    energy_pivot = results_df.pivot(
        index="medium_threshold", columns="green_threshold", values="energy_savings_pct"
    )

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    sns.set(style="whitegrid", font_scale=1.1)

    # Accuracy heatmap
    sns.heatmap(
        acc_pivot, annot=True, fmt=".3f", cmap="RdYlGn", ax=axes[0],
        cbar_kws={"label": "Accuracy"}
    )
    axes[0].set_title("Cascade Accuracy by Thresholds", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("Green Threshold")
    axes[0].set_ylabel("Medium Threshold")

    # Energy savings heatmap
    sns.heatmap(
        energy_pivot, annot=True, fmt=".1f", cmap="YlGn", ax=axes[1],
        cbar_kws={"label": "Energy Savings (%)"}
    )
    axes[1].set_title("Energy Savings by Thresholds", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("Green Threshold")
    axes[1].set_ylabel("Medium Threshold")

    plt.tight_layout()
    plt.savefig(output_path / "threshold_heatmap.png", dpi=300, bbox_inches="tight")
    print(f"✅ Saved: {output_path / 'threshold_heatmap.png'}")
    plt.close()


# ---------------------------------------------------------------------
def plot_pareto_frontier(results_df: pd.DataFrame, output_path: Path):
    """Plot Pareto frontier (Accuracy vs Energy Savings)."""
    print("\n📈 Plotting Pareto frontier...")

    fig, ax = plt.subplots(figsize=(10, 6))
    pareto_idx = []

    # Compute Pareto-optimal points
    for i, r in results_df.iterrows():
        dominated = any(
            (r2["accuracy"] >= r["accuracy"] and r2["energy_savings_pct"] > r["energy_savings_pct"])
            or (r2["accuracy"] > r["accuracy"] and r2["energy_savings_pct"] >= r["energy_savings_pct"])
            for _, r2 in results_df.iterrows()
        )
        if not dominated:
            pareto_idx.append(i)

    pareto_df = results_df.loc[pareto_idx].sort_values("energy_savings_pct")

    scatter = ax.scatter(
        results_df["energy_savings_pct"], results_df["accuracy"],
        c=results_df["green_usage"], cmap="viridis", alpha=0.6, s=50, label="All combinations"
    )

    ax.plot(
        pareto_df["energy_savings_pct"], pareto_df["accuracy"],
        "r-", lw=2, label="Pareto frontier"
    )

    ax.scatter(
        pareto_df["energy_savings_pct"], pareto_df["accuracy"],
        c="red", marker="*", s=100, edgecolors="black", label="Optimal"
    )

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Green Model Usage (%)")

    ax.set_xlabel("Energy Savings (%)")
    ax.set_ylabel("Accuracy")
    ax.set_title("Pareto Frontier: Accuracy vs Energy Savings", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path / "pareto_frontier.png", dpi=300, bbox_inches="tight")
    print(f"✅ Saved: {output_path / 'pareto_frontier.png'}")
    plt.close()

    return pareto_df


# ---------------------------------------------------------------------
def find_best_thresholds(results_df: pd.DataFrame, strategy="balanced"):
    """Find optimal thresholds by strategy."""
    if strategy == "max_accuracy":
        return results_df.loc[results_df["accuracy"].idxmax()]
    elif strategy == "max_energy":
        acc_cutoff = results_df["accuracy"].max() * 0.95
        subset = results_df[results_df["accuracy"] >= acc_cutoff]
        return subset.loc[subset["energy_savings_pct"].idxmax()]
    else:  # Balanced
        norm_acc = results_df["accuracy"] / results_df["accuracy"].max()
        norm_energy = results_df["energy_savings_pct"] / results_df["energy_savings_pct"].max()
        results_df = results_df.assign(score=0.7 * norm_acc + 0.3 * norm_energy)
        return results_df.loc[results_df["score"].idxmax()]


# ---------------------------------------------------------------------
def main():
    print("=" * 70)
    print("🔍 Cascade Threshold Optimization")
    print("=" * 70)

    # Load validation dataset
    print("\n📂 Loading validation set...")
    val_path = Config.PROCESSED_DATA_DIR / "val.csv"
    val_df = pd.read_csv(val_path)
    print(f"✅ Loaded {len(val_df)} samples.")

    # Auto sample (no input prompt)
    sample_size = min(1000, len(val_df))
    print(f"Using {sample_size} samples for optimization.\n")
    val_df = val_df.sample(n=sample_size, random_state=42)

    X_val = val_df["processed_text"].tolist()
    y_val = val_df["label"].tolist()

    # Load cascade classifier
    print("📦 Loading cascade classifier...")
    cascade = CascadeClassifier.load_models_and_create(
        Config.GREEN_MODEL_PATH, Config.MEDIUM_MODEL_PATH, Config.HEAVY_MODEL_PATH
    )

    # Threshold grid
    threshold_range = (0.50, 0.95)
    step = 0.05
    thresholds = np.arange(threshold_range[0], threshold_range[1] + step, step)
    print(f"\n🔢 Testing {len(thresholds)}×{len(thresholds)} = {len(thresholds)**2} combinations")

    results = []
    pbar = tqdm(total=len(thresholds) ** 2, desc="Grid search", ncols=80)

    for green_t in thresholds:
        for medium_t in thresholds:
            if medium_t > green_t:
                pbar.update(1)
                continue

            cascade.green_threshold = green_t
            cascade.medium_threshold = medium_t
            cascade.stats = {k: 0 for k in [
                "total_inferences", "green_used", "medium_used", "heavy_used",
                "green_correct", "medium_correct", "heavy_correct"
            ]}

            predictions = cascade.predict_batch(X_val, y_val, track_accuracy=True)
            y_pred = [p["prediction"] for p in predictions]
            acc = (y_val == pd.Series(y_pred)).mean()

            total = len(X_val)
            g, m, h = cascade.stats["green_used"], cascade.stats["medium_used"], cascade.stats["heavy_used"]
            cascade_energy = g * 1 + m * 10 + h * 30
            baseline_energy = total * 30
            energy_savings = (baseline_energy - cascade_energy) / baseline_energy * 100

            results.append({
                "green_threshold": green_t, "medium_threshold": medium_t,
                "accuracy": acc, "energy_savings_pct": energy_savings,
                "green_usage": g / total * 100, "medium_usage": m / total * 100,
                "heavy_usage": h / total * 100
            })
            pbar.update(1)
    pbar.close()

    results_df = pd.DataFrame(results)
    print("\n✅ Grid Search Complete!")

    # Select best thresholds by different strategies
    print("\n🎯 Optimal Thresholds:")
    strategies = {"Balanced": "balanced", "Max Accuracy": "max_accuracy", "Max Energy": "max_energy"}
    best_configs = {}

    for name, mode in strategies.items():
        best = find_best_thresholds(results_df, mode)
        best_configs[name] = best
        print(f"\n{name}:")
        print(f"   Green:  {best['green_threshold']:.2f}")
        print(f"   Medium: {best['medium_threshold']:.2f}")
        print(f"   Acc:    {best['accuracy']*100:.2f}%")
        print(f"   Energy: {best['energy_savings_pct']:.1f}%")

    # Save results
    out_dir = Config.PROJECT_ROOT / "optimization_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(out_dir / "grid_search_results.csv", index=False)
    print(f"\n💾 Saved results → {out_dir / 'grid_search_results.csv'}")

    plot_threshold_heatmap(results_df, out_dir)
    pareto_df = plot_pareto_frontier(results_df, out_dir)

    # Save best configs
    best_dict = {
        n: {
            "green_threshold": float(b["green_threshold"]),
            "medium_threshold": float(b["medium_threshold"]),
            "accuracy": float(b["accuracy"]),
            "energy_savings_pct": float(b["energy_savings_pct"]),
        }
        for n, b in best_configs.items()
    }

    with open(out_dir / "best_thresholds.json", "w") as f:
        json.dump(best_dict, f, indent=2)
    print(f"💾 Saved best configs → {out_dir / 'best_thresholds.json'}")

    # Apply balanced thresholds and save cascade config
    best_bal = best_configs["Balanced"]
    cascade.green_threshold = best_bal["green_threshold"]
    cascade.medium_threshold = best_bal["medium_threshold"]
    cascade.save(Config.MODELS_DIR / "cascade")

    print("\n" + "=" * 70)
    print("✅ OPTIMIZATION COMPLETE!")
    print("=" * 70)
    print(f"\n🌿 Recommended Thresholds (Balanced):")
    print(f"   Green:  {best_bal['green_threshold']:.2f}")
    print(f"   Medium: {best_bal['medium_threshold']:.2f}")
    print(f"   Accuracy: {best_bal['accuracy']*100:.2f}%")
    print(f"   Energy Savings: {best_bal['energy_savings_pct']:.1f}%")
    print(f"\nResults available in: {out_dir}/")


if __name__ == "__main__":
    main()
