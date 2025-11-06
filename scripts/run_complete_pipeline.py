# File: scripts/run_complete_pipeline.py
# Master script to run the complete GreenAI pipeline (Aligned Version)

import sys
import subprocess
from pathlib import Path
import time
import json
import os

# Add project root
sys.path.append(str(Path(__file__).parent.parent))
from src.config import Config


# ---------------------------------------------------------------------
def print_header(title):
    """Pretty phase headers"""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")


def run_command(command, description):
    """Run shell command and log results"""
    print(f"🚀 {description}...")
    print(f"   Command: {command}\n")

    start = time.time()
    try:
        subprocess.run(command, shell=True, check=True, text=True)
        print(f"\n✅ {description} complete! ({(time.time() - start) / 60:.1f} min)\n")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ {description} failed!\n   {e}")
        return False


# ---------------------------------------------------------------------
def check_prerequisites():
    """Verify project structure"""
    print_header("Checking Prerequisites")

    checks = {
        "Virtual environment": sys.prefix != sys.base_prefix,
        "Config module": (Config.PROJECT_ROOT / "src/config.py").exists(),
        "Database module": (Config.PROJECT_ROOT / "src/database.py").exists(),
        "Green model code": (Config.PROJECT_ROOT / "src/models/green_model.py").exists(),
        "Medium model code": (Config.PROJECT_ROOT / "src/models/medium_model.py").exists(),
        "Heavy model code": (Config.PROJECT_ROOT / "src/models/heavy_model.py").exists(),
        "Cascade code": (Config.PROJECT_ROOT / "src/models/cascade.py").exists(),
    }

    all_ok = True
    for name, ok in checks.items():
        print(f"{'✅' if ok else '❌'} {name}")
        if not ok:
            all_ok = False
    return all_ok


# ---------------------------------------------------------------------
def main():
    """Main pipeline orchestrator"""
    print_header("🌿 GreenAI Email Classifier — Complete Pipeline")

    print("Phases:")
    print("  1️⃣  Model Training (~1–3 hrs)")
    print("  2️⃣  Cascade Optimization (~20–60 min)\n")

    # Check structure
    if not check_prerequisites():
        print("\n❌ Missing required files. Please verify your setup.")
        return

    print("\n✅ All prerequisites found!")

    # Auto-run mode (no user input)
    auto_mode = os.getenv("AUTO_RUN", "true").lower() == "true"
    use_synthetic = True
    use_subset = True
    use_small_eval = True

    if not auto_mode:
        use_synthetic = input("\n📊 Use synthetic dataset? [y/n]: ").lower() == "y"
        use_subset = input("🔬 Use subset for training? [y/n]: ").lower() == "y"
        use_small_eval = input("⚡ Use small sample for evaluation? [y/n]: ").lower() == "y"

    total_start = time.time()

    # =====================
    # PHASE 1: DATA + MODELS
    # =====================
    print_header("PHASE 1 — Data Preparation & Model Training")

    cmd_data = (
        "python scripts/prepare_dataset.py --synthetic"
        if use_synthetic
        else "python scripts/prepare_dataset.py"
    )
    if not run_command(cmd_data, "Step 1: Preparing dataset"):
        print("❌ Aborted — dataset preparation failed.")
        return

    if use_subset:
        print("⚠️ Training on subset (20% of data).")

    if not run_command("python scripts/train_all_models.py",
                       "Step 2: Training Green, Medium, and Heavy models"):
        print("❌ Aborted — model training failed.")
        return

    print("🎉 Phase 1 complete.\n")

    # =====================
    # PHASE 2: CASCADE + OPTIMIZATION
    # =====================
    print_header("PHASE 2 — Cascade System & Energy Tracking")

    run_command("python src/models/cascade.py", "Step 3: Testing Cascade Module")

    print("\n🔍 Running Threshold Optimization...")
    run_command("python scripts/optimize_thresholds.py", "Step 4: Optimizing Thresholds")

    print("\n📊 Running Evaluation (Cascade vs Baseline)...")
    run_command("python scripts/evaluate_cascade.py", "Step 5: Full Evaluation")

    print("🎉 Phase 2 complete.\n")

    # =====================
    # FINAL SUMMARY
    # =====================
    total_time = time.time() - total_start
    print_header("🎊 PIPELINE COMPLETE")
    print(f"⏱️  Total runtime: {total_time/3600:.2f} hrs ({total_time/60:.1f} min)\n")

    print("📦 Artifacts Generated:")
    artifacts = {
        Config.PROCESSED_DATA_DIR / "train.csv": "Training data",
        Config.PROCESSED_DATA_DIR / "val.csv": "Validation data",
        Config.PROCESSED_DATA_DIR / "test.csv": "Test data",
        Config.GREEN_MODEL_PATH: "Green model",
        Config.MEDIUM_MODEL_PATH: "Medium model",
        Config.HEAVY_MODEL_PATH: "Heavy model",
        Config.MODELS_DIR / "cascade": "Cascade configuration",
        Config.PROJECT_ROOT / "optimization_results": "Threshold optimization",
        Config.PROJECT_ROOT / "evaluation_results": "Evaluation results",
        Config.PROJECT_ROOT / "greenai.db": "SQLite Database (if enabled)"
    }

    for path, desc in artifacts.items():
        print(f"{'✅' if path.exists() else '❌'} {str(path):60s} - {desc}")

    # Try to print summary results
    print("\n📊 Key Results:")
    results_path = Config.PROJECT_ROOT / "evaluation_results" / "results.json"

    try:
        if results_path.exists():
            with open(results_path) as f:
                res = json.load(f)
            cas_acc = res["cascade_metrics"]["accuracy"] * 100
            base_acc = res["baseline_metrics"]["accuracy"] * 100
            energy = res["energy_savings"]["energy_saved_percent"]
            co2 = res["energy_savings"]["co2_saved_percent"]

            print(f"   Cascade Accuracy:  {cas_acc:.2f}%")
            print(f"   Baseline Accuracy: {base_acc:.2f}%")
            print(f"   Accuracy Drop:     {base_acc - cas_acc:.2f}%")
            print(f"   Energy Saved:      {energy:.1f}%")
            print(f"   CO₂ Saved:         {co2:.1f}%")
        else:
            print("   ❌ Results file not found.")
    except Exception as e:
        print(f"   ⚠️ Could not load results: {e}")

    print("\n📈 Next Steps:")
    print("   1️⃣ Review: evaluation_results/evaluation_report.md")
    print("   2️⃣ View:   evaluation_results/evaluation_results.png")
    print("   3️⃣ Deploy: Proceed to Phase 3 (API + Dashboard)\n")

    print("🚀 To continue, type:  python scripts/run_phase3_api.py")


# ---------------------------------------------------------------------
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️ Pipeline interrupted — progress saved.")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
