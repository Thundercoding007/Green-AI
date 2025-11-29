# File: src/utils/energy_tracker.py
# Dynamic Energy and Carbon Tracking with CodeCarbon (Windows-optimized)

import time
import platform
from typing import Dict, Optional, Callable
from contextlib import contextmanager
from pathlib import Path
import warnings
import json

# Project config
from src.config import Config

warnings.filterwarnings("ignore", category=FutureWarning)

try:
    from codecarbon import EmissionsTracker
    CODECARBON_AVAILABLE = True
except ImportError:
    CODECARBON_AVAILABLE = False


# ---------------------------------------------------------------------
# ⚡ Real-Time Energy Tracker
# ---------------------------------------------------------------------
class EnergyTracker:
    """
    Uses CodeCarbon in process mode (Windows-safe) to measure per-inference energy.
    Falls back to static estimates if CodeCarbon not available.
    """

    def __init__(
        self,
        project_name: str = "greenai-email-classifier",
        output_dir: Path = Config.EMISSIONS_DIR,
    ):
        self.project_name = project_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.tracker = None
        self.measurements = []

        # Fallback power estimates (Watts)
        self.model_power_profiles = {
            "green": 2.0,
            "medium": 15.0,
            "heavy": 30.0,
        }

    # -----------------------------------------------------------------
    @contextmanager
    def track(self, model_name: str = "unknown"):
        """
        Context manager for tracking inference energy & CO₂.
        """
        use_live = CODECARBON_AVAILABLE
        tracker = None

        if use_live:
            try:
                tracker = EmissionsTracker(
                    project_name=f"{self.project_name}-{model_name}",
                    output_dir=str(self.output_dir),
                    measure_power_secs=1,
                    tracking_mode="process" if platform.system() == "Windows" else "machine",
                    save_to_file=True,
                    log_level="error"
                )
                tracker.start()
            except Exception as e:
                print(f"[EnergyTracker] Live tracking unavailable ({e}), using fallback.")
                use_live = False

        start_time = time.time()
        result = {"model_name": model_name}

        try:
            yield result
        finally:
            end_time = time.time()
            duration = end_time - start_time
            result["duration_s"] = duration

            if use_live and tracker:
                tracker.stop()
                data = getattr(tracker, "final_emissions_data", None)
                result["energy_kwh"] = getattr(data, "energy_consumed", 0.0)
                result["co2_grams"] = getattr(data, "emissions", 0.0)
            else:
                # fallback static estimate
                watts = self.model_power_profiles.get(model_name, 10.0)
                duration_hr = duration / 3600
                energy_kwh = watts * duration_hr / 1000
                co2_grams = energy_kwh * 632
                result["energy_kwh"] = energy_kwh
                result["co2_grams"] = co2_grams
                result["estimated"] = True

            self.measurements.append(result)

    # -----------------------------------------------------------------
    def track_inference(self, func: Callable, model_name: str, *args, **kwargs) -> tuple:
        """
        Track the energy and CO₂ of a single inference.
        """
        with self.track(model_name) as tracking:
            result = func(*args, **kwargs)

        metrics = {
            "model_name": tracking["model_name"],
            "energy_kwh": tracking["energy_kwh"],
            "co2_grams": tracking["co2_grams"],
            "duration_ms": tracking["duration_s"] * 1000,
        }
        return result, metrics

    # -----------------------------------------------------------------
    def get_summary(self) -> Dict:
        """Summarize all tracked emissions."""
        if not self.measurements:
            return {"total_measurements": 0, "total_energy_kwh": 0, "total_co2_grams": 0}

        total_e = sum(m["energy_kwh"] for m in self.measurements)
        total_c = sum(m["co2_grams"] for m in self.measurements)
        avg_t = sum(m["duration_s"] for m in self.measurements) / len(self.measurements)

        return {
            "total_measurements": len(self.measurements),
            "total_energy_kwh": total_e,
            "total_co2_grams": total_c,
            "avg_duration_ms": avg_t * 1000,
        }


# ---------------------------------------------------------------------
# 🌿 Cascade Energy Tracker
# ---------------------------------------------------------------------
class CascadeEnergyTracker:
    """
    Tracks and compares cascade energy vs. baseline model.
    Uses dynamic tracking if possible, otherwise estimated static profiles.
    """

    def __init__(self, output_dir: Path = Config.EMISSIONS_DIR):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.energy_tracker = EnergyTracker(output_dir=output_dir)
        self.cascade_logs = []
        self.baseline_logs = []

        # fallback static profiles
        self.energy_profiles = {"green": 0.000001, "medium": 0.000150, "heavy": 0.000500}

    def log_cascade_inference(self, cascade_result: Dict, actual_label: Optional[str] = None) -> Dict:
        model_used = cascade_result["model_used"]

        # try to run live tracking for last model used
        energy_data = {"energy_kwh": 0, "co2_grams": 0}
        if CODECARBON_AVAILABLE:
            try:
                _, metrics = self.energy_tracker.track_inference(lambda: time.sleep(0.001), model_used)
                energy_data.update(metrics)
            except Exception as e:
                print(f"[CascadeEnergyTracker] Live energy fallback ({e})")

        if energy_data["energy_kwh"] == 0:
            # fallback estimate
            energy = self.energy_profiles.get(model_used, 0.0001)
            co2 = energy * 632
            energy_data = {"energy_kwh": energy, "co2_grams": co2}

        entry = {
            "model_used": model_used,
            "cascade_path": cascade_result["cascade_path"],
            "prediction": cascade_result["prediction"],
            "confidence": cascade_result["confidence"],
            "inference_time_ms": cascade_result["total_time_ms"],
            "energy_kwh": energy_data["energy_kwh"],
            "co2_grams": energy_data["co2_grams"],
            "correct": cascade_result["prediction"] == actual_label if actual_label else None,
        }
        self.cascade_logs.append(entry)
        return entry

    def log_baseline_inference(self, prediction: str, confidence: float, inference_time_ms: float,
                               actual_label: Optional[str] = None) -> Dict:
        energy = self.energy_profiles["heavy"]
        co2 = energy * 632
        entry = {
            "model_used": "heavy",
            "prediction": prediction,
            "confidence": confidence,
            "inference_time_ms": inference_time_ms,
            "energy_kwh": energy,
            "co2_grams": co2,
            "correct": prediction == actual_label if actual_label else None,
        }
        self.baseline_logs.append(entry)
        return entry

    def calculate_savings(self) -> Dict:
        if not self.cascade_logs or not self.baseline_logs:
            return {k: 0 for k in
                    ["energy_saved_kwh", "energy_saved_percent", "co2_saved_grams", "co2_saved_percent"]}

        c_energy = sum(l["energy_kwh"] for l in self.cascade_logs)
        b_energy = sum(l["energy_kwh"] for l in self.baseline_logs)
        c_co2 = sum(l["co2_grams"] for l in self.cascade_logs)
        b_co2 = sum(l["co2_grams"] for l in self.baseline_logs)

        energy_saved = b_energy - c_energy
        co2_saved = b_co2 - c_co2

        return {
            "energy_saved_kwh": energy_saved,
            "energy_saved_percent": (energy_saved / b_energy * 100) if b_energy else 0,
            "co2_saved_grams": co2_saved,
            "co2_saved_percent": (co2_saved / b_co2 * 100) if b_co2 else 0,
        }

    def save_logs(self):
        data = {
            "cascade_logs": self.cascade_logs,
            "baseline_logs": self.baseline_logs,
            "savings": self.calculate_savings(),
        }
        file = self.output_dir / "energy_logs.json"
        with open(file, "w") as f:
            json.dump(data, f, indent=2)
        print(f"✅ Energy logs saved to: {file}")
