# File: src/utils/energy_tracker.py
# Energy and Carbon Tracking Utilities (Aligned with project standards)

import time
from typing import Dict, Optional, Callable
from contextlib import contextmanager
from pathlib import Path
import warnings
import json

# Project config
from src.config import Config

warnings.filterwarnings("ignore", category=FutureWarning)

try:
    from codecarbon import EmissionsTracker, OfflineEmissionsTracker
    CODECARBON_AVAILABLE = True
except ImportError:
    CODECARBON_AVAILABLE = False


class EnergyTracker:
    """
    Wrapper around CodeCarbon for real or estimated energy tracking.
    """

    def __init__(
        self,
        project_name: str = "greenai-email-classifier",
        output_dir: Path = Config.EMISSIONS_DIR,
        country_iso_code: str = "IND",
        offline: bool = True,
    ):
        self.project_name = project_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.country_iso_code = country_iso_code
        self.offline = offline

        # Power profiles (Watts per inference type)
        self.model_power_profiles = {
            "green": 2.0,   # lightweight CPU
            "medium": 15.0, # GPU or hybrid
            "heavy": 30.0,  # heavy GPU transformer
        }

        self.measurements = []

    # -----------------------------------------------------------------
    @contextmanager
    def track(self, model_name: str = "unknown"):
        """
        Context manager for real-time tracking (if CodeCarbon is available).
        """
        if CODECARBON_AVAILABLE:
            tracker_cls = OfflineEmissionsTracker if self.offline else EmissionsTracker
            tracker = tracker_cls(
                project_name=self.project_name,
                output_dir=str(self.output_dir),
                country_iso_code=self.country_iso_code if self.offline else None,
                log_level="error",
            )
            tracker.start()
        else:
            tracker = None

        start_time = time.time()
        result = {"model_name": model_name}

        try:
            yield result
        finally:
            end_time = time.time()
            duration = end_time - start_time
            emissions = tracker.stop() if tracker else None

            result.update({
                "duration_s": duration,
                "energy_kwh": emissions or 0,
                "co2_grams": (emissions or 0) * 1000,
            })
            self.measurements.append(result)

    # -----------------------------------------------------------------
    def track_inference(self, func: Callable, model_name: str, *args, **kwargs) -> tuple:
        """
        Track the energy and CO₂ of a specific inference.
        Falls back to estimate if CodeCarbon not available.
        """
        if CODECARBON_AVAILABLE:
            with self.track(model_name) as tracking:
                result = func(*args, **kwargs)
            metrics = {
                "model_name": tracking["model_name"],
                "energy_kwh": tracking["energy_kwh"],
                "co2_grams": tracking["co2_grams"],
                "duration_ms": tracking["duration_s"] * 1000,
            }
        else:
            start = time.time()
            result = func(*args, **kwargs)
            duration_ms = (time.time() - start) * 1000
            metrics = self.estimate_energy(model_name, duration_ms)

        return result, metrics

    # -----------------------------------------------------------------
    def estimate_energy(self, model_name: str, duration_ms: float) -> Dict:
        """
        Rough energy estimate based on power draw & inference duration.
        """
        watts = self.model_power_profiles.get(model_name, 10.0)
        duration_hr = duration_ms / (1000 * 3600)
        energy_kwh = watts * duration_hr / 1000
        grid_intensity = 632  # gCO₂/kWh (India)
        co2_grams = energy_kwh * grid_intensity

        return {
            "model_name": model_name,
            "energy_kwh": energy_kwh,
            "co2_grams": co2_grams,
            "duration_ms": duration_ms,
            "estimated": True,
            "power_watts": watts,
        }

    # -----------------------------------------------------------------
    def get_summary(self) -> Dict:
        """Summarize all tracked emissions."""
        if not self.measurements:
            return {
                "total_measurements": 0,
                "total_energy_kwh": 0,
                "total_co2_grams": 0,
                "avg_duration_ms": 0,
            }

        total_e = sum(m.get("energy_kwh", 0) for m in self.measurements)
        total_c = sum(m.get("co2_grams", 0) for m in self.measurements)
        total_d = sum(m.get("duration_s", 0) for m in self.measurements)

        return {
            "total_measurements": len(self.measurements),
            "total_energy_kwh": total_e,
            "total_co2_grams": total_c,
            "avg_duration_ms": (total_d / len(self.measurements)) * 1000,
        }


# ---------------------------------------------------------------------
# ⚡ Cascade Energy Tracker
# ---------------------------------------------------------------------
class CascadeEnergyTracker:
    """
    Tracks energy and carbon footprint for CascadeClassifier runs.
    """

    def __init__(self, output_dir: Path = Config.EMISSIONS_DIR):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Energy cost per inference type (kWh)
        self.energy_profiles = {
            "green": 0.000001,  # 1 µWh
            "medium": 0.000150, # 150 µWh
            "heavy": 0.000500,  # 500 µWh
        }

        self.cascade_logs = []
        self.baseline_logs = []

    # -----------------------------------------------------------------
    def log_cascade_inference(self, cascade_result: Dict, actual_label: Optional[str] = None) -> Dict:
        """Record energy and CO₂ for a cascade decision."""
        model_used = cascade_result["model_used"]
        cascade_path = cascade_result["cascade_path"].split("->")
        energy = sum(self.energy_profiles.get(m, 0.0001) for m in cascade_path)
        co2 = energy * 632

        entry = {
            "model_used": model_used,
            "cascade_path": cascade_result["cascade_path"],
            "prediction": cascade_result["prediction"],
            "confidence": cascade_result["confidence"],
            "inference_time_ms": cascade_result["total_time_ms"],
            "energy_kwh": energy,
            "co2_grams": co2,
            "correct": cascade_result["prediction"] == actual_label if actual_label else None,
        }
        self.cascade_logs.append(entry)
        return entry

    # -----------------------------------------------------------------
    def log_baseline_inference(self, prediction: str, confidence: float, inference_time_ms: float,
                               actual_label: Optional[str] = None) -> Dict:
        """Baseline heavy-only inference log."""
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

    # -----------------------------------------------------------------
    def calculate_savings(self) -> Dict:
        """Compute relative savings between cascade and heavy baseline."""
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
            "cascade_energy_kwh": c_energy,
            "baseline_energy_kwh": b_energy,
            "cascade_co2_grams": c_co2,
            "baseline_co2_grams": b_co2,
        }

    # -----------------------------------------------------------------
    def get_model_distribution(self) -> Dict:
        """Return usage percentage of each tier."""
        total = len(self.cascade_logs)
        if not total:
            return {}

        def pct(model): return sum(1 for l in self.cascade_logs if l["model_used"] == model) / total * 100

        return {m: pct(m) for m in ["green", "medium", "heavy"]}

    # -----------------------------------------------------------------
    def save_logs(self, path: Path = None):
        """Save all energy logs and summaries."""
        if path is None:
            path = self.output_dir
        path.mkdir(parents=True, exist_ok=True)

        data = {
            "cascade_logs": self.cascade_logs,
            "baseline_logs": self.baseline_logs,
            "savings": self.calculate_savings(),
            "model_distribution": self.get_model_distribution(),
        }

        with open(path / "energy_logs.json", "w") as f:
            json.dump(data, f, indent=2)
        print(f"✅ Energy logs saved to: {path / 'energy_logs.json'}")


# ---------------------------------------------------------------------
# 🧪 Self-Test
# ---------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 70)
    print("⚡ Testing Energy Tracking")
    print("=" * 70)

    tracker = EnergyTracker()
    print("\n1️⃣ Testing EnergyTracker (estimation mode)...")

    def dummy_inference():
        time.sleep(0.01)
        return "work"

    result, metrics = tracker.track_inference(dummy_inference, "green")
    print(f"   ✅ Result: {result}, {metrics['energy_kwh']:.9e} kWh, {metrics['co2_grams']:.6f} g")

    print("\n2️⃣ Testing CascadeEnergyTracker...")
    cascade_tracker = CascadeEnergyTracker()

    fake_cascade = {
        "prediction": "work",
        "confidence": 0.92,
        "model_used": "green",
        "cascade_path": "green->medium",
        "total_time_ms": 8.5,
    }

    c_log = cascade_tracker.log_cascade_inference(fake_cascade, actual_label="work")
    b_log = cascade_tracker.log_baseline_inference("work", 0.93, 280.0, "work")

    print(f"   ⚙️ Cascade Energy: {c_log['energy_kwh']:.9f} kWh, CO₂: {c_log['co2_grams']:.6f} g")
    print(f"   ⚙️ Baseline Energy: {b_log['energy_kwh']:.9f} kWh, CO₂: {b_log['co2_grams']:.6f} g")

    savings = cascade_tracker.calculate_savings()
    print(f"\n💰 Savings: {savings['energy_saved_percent']:.2f}% energy, {savings['co2_saved_percent']:.2f}% CO₂")

    cascade_tracker.save_logs()
    print("\n✅ Energy tracking test complete!\n")
