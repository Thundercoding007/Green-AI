#!/usr/bin/env bash
# ==========================================================
# 🌱 GreenAI Emission Monitoring Script (Windows Compatible)
# ----------------------------------------------------------
# Measures energy usage in:
#   (1) Baseline Mode  – when app is NOT running
#   (2) Application Mode – when app IS running
# ==========================================================

set -e

# -------------------------------
# 🧩 Parse Arguments
# -------------------------------
APP_RUN=""
TEAM_NAME=""
RUN_SECONDS=""
DEBUG="false"

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --app-run) APP_RUN="$2" ;;
        --team-name) TEAM_NAME="$2" ;;
        --run-seconds) RUN_SECONDS="$2" ;;
        --debug) DEBUG="$2" ;;
    esac
    shift
done

if [[ -z "$APP_RUN" || -z "$TEAM_NAME" ]]; then
    echo "❌ Usage: ./emission_calculation_windows.sh --app-run [true|false] --team-name <team_name> [--run-seconds <seconds>] [--debug true|false]"
    exit 1
fi

if [[ "$APP_RUN" == "true" && -z "$RUN_SECONDS" ]]; then
    echo "❌ Missing required parameter: --run-seconds for app-run=true"
    exit 1
fi

# Default baseline duration = 3600 sec (1 hr)
if [[ "$APP_RUN" == "false" ]]; then
    RUN_SECONDS=3600
fi

# -------------------------------
# 📁 Setup Paths
# -------------------------------
LOG_DIR="./emissions_logs"
mkdir -p "$LOG_DIR"

VENV_DIR="./emissions_env"
OUTPUT_FILE="$LOG_DIR/monitoring_app_run_${APP_RUN}.csv"

# -------------------------------
# 🐍 Setup Python Environment
# -------------------------------
if [[ ! -d "$VENV_DIR" ]]; then
    echo "🔧 Creating Python virtual environment..."
    python -m venv "$VENV_DIR"
fi

# ✅ Windows fix: activate from 'Scripts' instead of 'bin'
if [[ -f "$VENV_DIR/Scripts/activate" ]]; then
    source "$VENV_DIR/Scripts/activate"
else
    echo "❌ Could not find activation script at $VENV_DIR/Scripts/activate"
    exit 1
fi

echo "📦 Installing dependencies..."
pip install --quiet psutil pandas tqdm

# -------------------------------
# ⚡ Start Monitoring
# -------------------------------
echo "🚀 Starting energy monitoring for ${RUN_SECONDS}s..."
echo "🧠 Mode: app_run=${APP_RUN}, team_name=${TEAM_NAME}"

export RUN_SECONDS TEAM_NAME APP_RUN OUTPUT_FILE

python <<'EOF'
import psutil, time, csv, os
from datetime import datetime
from tqdm import tqdm

duration = int(os.getenv("RUN_SECONDS", "60"))
team_name = os.getenv("TEAM_NAME", "Unknown")
app_run = os.getenv("APP_RUN", "false")
log_file = os.getenv("OUTPUT_FILE", "./monitor.csv")

fields = ["timestamp", "duration", "cpu_percent", "memory_percent", "energy_kwh", "team_name", "app_run"]
start_time = datetime.now()
rows = []

def get_energy(cpu, mem):
    return (cpu*0.02 + mem*0.005) / 3600.0  # simplified kWh proxy

print(f"⏳ Monitoring for {duration}s ...")

for _ in tqdm(range(duration), desc="Monitoring Progress"):
    cpu = psutil.cpu_percent(interval=1)
    mem = psutil.virtual_memory().percent
    energy = get_energy(cpu, mem)
    rows.append({
        "timestamp": datetime.now().isoformat(),
        "duration": (datetime.now() - start_time).seconds,
        "cpu_percent": cpu,
        "memory_percent": mem,
        "energy_kwh": energy,
        "team_name": team_name,
        "app_run": app_run
    })

os.makedirs(os.path.dirname(log_file), exist_ok=True)
with open(log_file, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)

print(f"\n✅ Monitoring complete! Data saved to {log_file}")
EOF

deactivate

echo ""
echo "════════════════════════════════════════════════════════════"
echo "✅ Process completed successfully!"
echo "  Output file: $OUTPUT_FILE"
echo "  Team Name: $TEAM_NAME"
echo "  App Running: $APP_RUN"
echo "  Duration: $RUN_SECONDS seconds"
echo "════════════════════════════════════════════════════════════"
