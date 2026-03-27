# File: src/dashboard/app.py
# Streamlit Dashboard for GreenAI Email Classifier (rewritten)
# - Replaces manual sliders with read-only router-info display
# - Shows per-class thresholds, temps, validation stats
# - Keeps classify, analytics, and dashboard views intact
# - ❗ CascadeClassifier fully removed — now uses GreenRouter

import streamlit as st
import pandas as pd
import plotly.express as px
import requests
import time
from datetime import datetime, timedelta
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.config import Config

# Optional DB imports (safe fallback)
try:
    from src.database import (
        SessionLocal,
        InferenceLog,
        get_model_statistics,
        calculate_energy_savings,
    )

    DB_AVAILABLE = True
except Exception:
    DB_AVAILABLE = False


# -------------------------------------------------------------
# Page config
# -------------------------------------------------------------
st.set_page_config(
    page_title="🌿 GreenAI Email Classifier",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -------------------------------------------------------------
# Custom CSS
# -------------------------------------------------------------
st.markdown(
    """
<style>
.main-header {
    font-size: 3rem;
    font-weight: bold;
    color: #2ecc71;
    text-align: center;
    margin-bottom: 1rem;
}
div[data-testid="stMetricValue"], div[data-testid="stMetricDelta"] {
    color: #ffffff !important;
    font-weight: 600 !important;
}
div[data-testid="stMetric"] {
    background-color: rgba(40, 40, 40, 0.9) !important;
    border: 1px solid #2ecc71;
    border-radius: 0.75rem !important;
    padding: 1rem !important;
}
div[data-testid="stDataFrame"] {
    background-color: #1e1e1e !important;
    color: #e0e0e0 !important;
    border-radius: 0.5rem;
}
thead, th, td {
    color: #ddd !important;
    background-color: #1b1b1b !important;
}
button[kind="primary"] {
    background-color: #2ecc71 !important;
    color: white !important;
    border-radius: 0.5rem !important;
    border: none !important;
    font-weight: 600;
}
</style>
""",
    unsafe_allow_html=True,
)

# -------------------------------------------------------------
# API Configuration
# -------------------------------------------------------------
API_BASE_URL = os.getenv(
    "API_BASE_URL",
    f"http://{Config.API_HOST}:{Config.API_PORT}"
)

if "0.0.0.0" in API_BASE_URL:
    API_BASE_URL = API_BASE_URL.replace("0.0.0.0", "localhost")


# -------------------------------------------------------------
# Helper functions (unchanged)
# -------------------------------------------------------------
@st.cache_data(ttl=60)
def get_api_health():
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.json()
    except:
        return None


@st.cache_data(ttl=30)
def get_api_stats():
    try:
        response = requests.get(f"{API_BASE_URL}/stats", timeout=5)
        return response.json()
    except:
        return None


@st.cache_data(ttl=60)
def get_api_config():
    try:
        response = requests.get(f"{API_BASE_URL}/config", timeout=5)
        return response.json()
    except:
        return None


@st.cache_data(ttl=60)
def get_router_info():
    """➜ NEW: replaced cascade-info with router-info"""
    try:
        response = requests.get(f"{API_BASE_URL}/config/router-info", timeout=5)
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None


def load_database_stats():
    if not DB_AVAILABLE:
        st.warning(
            "⚠️ Database not available. Some analytics features are disabled."
        )
        return None

    db = SessionLocal()
    try:
        logs = (
            db.query(InferenceLog)
            .order_by(InferenceLog.timestamp.desc())
            .limit(2000)
            .all()
        )

        if not logs:
            return None

        df = pd.DataFrame(
            [
                {
                    "timestamp": log.timestamp,
                    "model_used": log.model_used,
                    "prediction": log.predicted_class,
                    "confidence": log.confidence,
                    "energy_kwh": log.energy_kwh,
                    "co2_grams": log.co2_grams,
                    "inference_time_ms": log.inference_time_ms,
                    "correct": log.correct,
                    # cascade_path removed — router has no cascade
                    "cascade_path": "N/A",
                }
                for log in logs
            ]
        )

        return df

    finally:
        db.close()


def classify_email(text: str):
    try:
        response = requests.post(
            f"{API_BASE_URL}/classify",
            json={"text": text, "track_energy": True},
            timeout=30,
        )
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": response.json().get("detail", "Unknown error")}
    except Exception as e:
        return {"error": str(e)}


def format_energy_kwh(kwh: float):
    if kwh is None:
        return "N/A"

    if kwh < 0.001:
        return f"{kwh * 1_000_000:.2f} µWh"
    elif kwh < 1:
        return f"{kwh * 1_000:.4f} Wh"
    else:
        return f"{kwh:.6f} kWh"


# -------------------------------------------------------------
# Main Entry Point (UNCHANGED)
# -------------------------------------------------------------
def main():
    st.markdown(
        '<p class="main-header">🌿 GreenAI Email Classifier</p>',
        unsafe_allow_html=True,
    )
    st.markdown("### Sustainable AI-Powered Email Classification")

    # Sidebar
    with st.sidebar:
        st.image(
            "https://via.placeholder.com/150x150/2ecc71/ffffff?text=GreenAI",
            use_column_width=True,
        )

        st.markdown("---")
        st.subheader("🏥 System Health")

        health = get_api_health()
        if health:
            status_color = "🟢" if health["status"] == "healthy" else "🟡"
            st.markdown(
                f"{status_color} **Status:** {health['status'].upper()}"
            )
            st.markdown(
                f"✅ Models Loaded: {'Yes' if health['models_loaded'] else 'No'}"
            )

            # ⬇️ REPLACED
            st.markdown(
                f"✅ Router Ready: {'Yes' if health.get('router_ready', False) else 'No'}"
            )

            st.markdown(
                f"✅ Database: {'Yes' if health['database_connected'] else 'No'}"
            )
        else:
            st.markdown("🔴 **Status:** API Offline")
            st.error("Cannot connect to API. Please start the API server.")

        st.markdown("---")
        st.subheader("📍 Navigation")

        page = st.radio(
            "Select Page",
            [
                "🏠 Dashboard",
                "🎯 Classify Email",
                "📊 Analytics",
                "⚙️ Configuration",
            ],
            label_visibility="collapsed",
        )

        st.markdown("---")

        stats = get_api_stats()
        if stats:
            st.subheader("📈 Quick Stats")
            st.metric("Total Inferences", f"{stats.get('total_inferences', 0):,}")
            if stats.get("energy_saved_percent"):
                st.metric(
                    "Energy Saved",
                    f"{stats['energy_saved_percent']:.1f}%",
                )

    # Page Routing
    if page == "🏠 Dashboard":
        show_dashboard()
    elif page == "🎯 Classify Email":
        show_classifier()
    elif page == "📊 Analytics":
        show_analytics()
    elif page == "⚙️ Configuration":
        show_configuration()


# -------------------------------------------------------------
# PAGE: Dashboard (UNCHANGED except cascade_path -> N/A)
# -------------------------------------------------------------
def show_dashboard():
    st.header("📊 System Overview")

    df = load_database_stats()
    stats = get_api_stats()

    if df is None or len(df) == 0:
        st.info(
            "📭 No inference data yet. Start classifying emails to see statistics!"
        )
        return

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Total Inferences",
            f"{len(df):,}",
            delta=f"+{len(df[df['timestamp'] > datetime.now() - timedelta(hours=24)])} today",
        )

    with col2:
        avg_conf = df["confidence"].mean() * 100
        st.metric("Avg Confidence", f"{avg_conf:.1f}%")

    with col3:
        if stats and stats.get("energy_saved_percent"):
            st.metric(
                "Energy Saved",
                f"{stats['energy_saved_percent']:.1f}%",
                delta="vs baseline",
            )
        else:
            st.metric("Energy Saved", "N/A")

    with col4:
        total_co2 = df["co2_grams"].sum()
        st.metric("CO₂ Emissions", f"{total_co2:.6f}g")

    st.markdown("---")

    # Model usage
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🎯 Model Usage Distribution")
        model_counts = df["model_used"].value_counts()
        fig = px.pie(
            values=model_counts.values,
            names=model_counts.index,
            hole=0.4,
        )
        fig.update_traces(textposition="inside", textinfo="percent+label")
        st.plotly_chart(fig, use_container_width=True)

    # Class distribution
    with col2:
        st.subheader("📈 Classification Distribution")
        class_counts = df["prediction"].value_counts()
        fig = px.bar(
            x=class_counts.index,
            y=class_counts.values,
            labels={"x": "Class", "y": "Count"},
        )
        st.plotly_chart(fig, use_container_width=True)

    # Energy/time trends
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("⚡ Energy Consumption Over Time")
        df["hour"] = pd.to_datetime(df["timestamp"]).dt.floor("H")
        hourly = df.groupby("hour")["energy_kwh"].sum().reset_index()
        fig = px.line(
            hourly,
            x="hour",
            y="energy_kwh",
            labels={"hour": "Time", "energy_kwh": "Energy (kWh)"},
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("🚀 Inference Time by Model")
        fig = px.box(df, x="model_used", y="inference_time_ms")
        st.plotly_chart(fig, use_container_width=True)

    # Recent Inferences
    st.subheader("📝 Recent Inferences")
    recent_df = df.head(10)[
        [
            "timestamp",
            "prediction",
            "confidence",
            "model_used",
            "inference_time_ms",
            "energy_kwh",
        ]
    ]

    recent_df["confidence"] = recent_df["confidence"].apply(
        lambda x: f"{x * 100:.1f}%"
    )
    recent_df["inference_time_ms"] = recent_df[
        "inference_time_ms"
    ].apply(lambda x: f"{x:.2f}ms")
    recent_df["energy_kwh"] = recent_df["energy_kwh"].apply(
        format_energy_kwh
    )

    st.dataframe(recent_df, use_container_width=True)


# -------------------------------------------------------------
# PAGE: Classifier (cascade removed)
# -------------------------------------------------------------
def show_classifier():
    st.header("🎯 Email Classification")

    health = get_api_health()
    if not health or health["status"] != "healthy":
        st.error("⚠️ API is not available. Please start the API server first.")
        return

    tab1, tab2 = st.tabs(["📝 Single Email", "📚 Batch Upload"])

    # ----------------------------------------
    # Tab 1: Single Email Classification
    # ----------------------------------------
    with tab1:
        st.subheader("Classify a Single Email")

        samples = {
            "Meeting Request": (
                "Hi team, let's schedule a meeting tomorrow "
                "at 2pm to discuss the quarterly results."
            ),
            "Promotional": (
                "HUGE SALE! Get 70% off on all products. "
                "Limited time offer. Shop now!"
            ),
            "Support Ticket": (
                "Your support ticket #12345 has been received. "
                "Our team will respond within 24 hours."
            ),
            "Personal": (
                "Hey! Want to grab coffee this weekend? "
                "It's been too long since we caught up."
            ),
        }

        selected_sample = st.selectbox(
            "Try a sample email:",
            ["Custom"] + list(samples.keys()),
        )

        default_text = (
            samples[selected_sample]
            if selected_sample != "Custom"
            and st.button("Load Sample")
            else ""
        )

        email_text = st.text_area("Email Text:", value=default_text, height=200)

        if st.button("🚀 Classify", type="primary"):
            if len(email_text) < 10:
                st.error("Please enter at least 10 characters.")
            else:
                with st.spinner("🔄 Classifying..."):
                    result = classify_email(email_text)

                    if "error" in result:
                        st.error(f"❌ Error: {result['error']}")
                        return

                    st.success("✅ Classification Complete!")

                    # ---- High-level metrics (prediction / chosen model) ----
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Prediction", result["prediction"].upper())
                    col2.metric("Confidence", f"{result['confidence'] * 100:.1f}%")
                    col3.metric("Model Used", result["model_used"].upper())
                    col4.metric("Time", f"{result['inference_time_ms']:.2f}ms")

                    # ---- Per-model confidences (Option C, now dynamic) ----
                    per_model_conf = result.get("per_model_confidences") or {}
                    router_cfg = get_router_info() or {}
                    thresholds_lr = router_cfg.get("thresholds_lr", {})
                    thresholds_med = router_cfg.get("thresholds_med", {})

                    selected_model = result["model_used"].lower()

                    # helper to get scalar threshold for the predicted class index (if possible)
                    def _get_threshold_for_model(model_name, chosen_label_idx=None):
                        tmap = thresholds_lr if model_name == "green" else thresholds_med
                        if chosen_label_idx is not None:
                            return float(tmap.get(str(chosen_label_idx), 0.5))
                        return float(tmap.get("0", 0.5))

                    # Determine chosen label index
                    chosen_label_idx = None
                    try:
                        labels = router_cfg.get("classes", Config.CLASSES)
                        if result["prediction"] in labels:
                            chosen_label_idx = labels.index(result["prediction"])
                    except Exception:
                        chosen_label_idx = None

                    # ---------------------------
                    #   Dynamic display logic
                    # ---------------------------

                    st.subheader("🔍 Per-model Confidence & Thresholds")

                    # Only show models UP TO the one used
                    display_models = {
                        "green": ["green"],
                        "medium": ["green", "medium"],
                        "heavy": ["green", "medium", "heavy"],
                    }[selected_model]

                    pm_cols = st.columns(len(display_models))

                    for i, model_key in enumerate(display_models):
                        with pm_cols[i]:
                            conf = per_model_conf.get(model_key)
                            if conf is None:
                                st.write(model_key.capitalize())
                                st.write("No data")
                                continue

                            thr = _get_threshold_for_model(model_key, chosen_label_idx)
                            passed = conf >= thr
                            status = "ACCEPT" if passed else "REJECT"

                            st.metric(
                                f"{model_key.capitalize()} Confidence",
                                f"{conf*100:.1f}%",
                                delta=f"{status} vs {thr*100:.1f}%"
                            )

                    # ---- Expanded detailed info (probabilities + reason) ----
                    with st.expander("📊 Detailed Results"):
                        st.write(f"- Selected Model: **{result['model_used'].upper()}**")
                        st.write(f"- Cascade Path: {result.get('cascade_path', '')}")
                        st.write(f"- Energy: {format_energy_kwh(result.get('energy_kwh'))}" if result.get('energy_kwh') else "- Energy: N/A")
                        st.write(f"- CO₂: {result.get('co2_grams', 0):.6f} g" if result.get('co2_grams') is not None else "- CO₂: N/A")

                        # Per-model probability bars
                        per_model_probs = result.get("per_model_probs") or {}
                        for model_key in display_models:
                            if model_key not in per_model_probs:
                                continue

                            st.write(f"**{model_key.capitalize()} probabilities**")
                            prob_map = per_model_probs[model_key]
                            probs_df = pd.DataFrame({
                                "Class": list(prob_map.keys()),
                                "Probability": list(prob_map.values())
                            }).sort_values("Probability", ascending=False)

                            # normalize if needed
                            if probs_df["Probability"].max() > 1:
                                probs_df["Probability"] = probs_df["Probability"] / 100.0

                            fig = px.bar(
                                probs_df, x="Class", y="Probability",
                                text=probs_df["Probability"].apply(lambda x: f"{x*100:.1f}%")
                            )
                            fig.update_traces(textposition="outside")
                            fig.update_layout(yaxis_tickformat=".0%")
                            st.plotly_chart(fig, use_container_width=True)

                        # Explanation
                        st.markdown("**Why this model was chosen:**")
                        st.write("The router evaluates models in order (Green → Medium → Heavy).")
                        st.write("Each model’s confidence is compared with a per-class threshold.")
                        st.write("If confidence ≥ threshold → prediction is accepted. Otherwise escalates.")


    # ----------------------------------------
    # Tab 2: Batch Upload (unchanged)
    # ----------------------------------------
    with tab2:
        st.subheader("Batch Classification")

        uploaded_file = st.file_uploader(
            "Choose a CSV file", type=["csv"]
        )

        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file)

                if "text" not in df.columns:
                    st.error("CSV must contain a 'text' column")
                    return

                st.write(f"Loaded {len(df)} emails")

                if st.button("🚀 Classify All", type="primary"):
                    progress = st.progress(0)
                    results = []

                    for i, text in enumerate(df["text"]):
                        result = classify_email(str(text))
                        results.append(result)
                        progress.progress((i + 1) / len(df))

                    df["prediction"] = [
                        r.get("prediction", "error") for r in results
                    ]
                    df["confidence"] = [
                        r.get("confidence", 0) for r in results
                    ]
                    df["model_used"] = [
                        r.get("model_used", "error") for r in results
                    ]

                    st.success("✅ Batch classification complete!")
                    st.dataframe(df, use_container_width=True)

                    st.download_button(
                        "📥 Download Results",
                        df.to_csv(index=False),
                        "classification_results.csv",
                        "text/csv",
                    )

            except Exception as e:
                st.error(f"Error processing file: {e}")


# -------------------------------------------------------------
# PAGE: Analytics (unchanged)
# -------------------------------------------------------------
def show_analytics():
    st.header("📊 Advanced Analytics")

    df = load_database_stats()

    if df is None or len(df) == 0:
        st.info("📭 No data available for analytics yet.")
        return

    date_range = st.date_input(
        "Date Range",
        value=(
            df["timestamp"].min().date(),
            df["timestamp"].max().date(),
        ),
    )

    model_filter = st.multiselect(
        "Models",
        options=["green", "medium", "heavy"],
        default=["green", "medium", "heavy"],
    )

    class_filter = st.multiselect(
        "Classes",
        options=df["prediction"].unique().tolist(),
        default=df["prediction"].unique().tolist(),
    )

    filtered_df = df[
        (df["model_used"].isin(model_filter))
        & (df["prediction"].isin(class_filter))
    ]

    st.metric("Filtered Records", f"{len(filtered_df):,}")

    tab1, _, _ = st.tabs(["📈 Trends", "🎯 Performance", "⚡ Efficiency"])

    with tab1:
        filtered_df["date"] = pd.to_datetime(
            filtered_df["timestamp"]
        ).dt.date

        daily = filtered_df.groupby("date").size().reset_index(
            name="count"
        )

        fig = px.line(
            daily,
            x="date",
            y="count",
            labels={"date": "Date", "count": "Inferences"},
        )

        st.plotly_chart(fig, use_container_width=True)


# -------------------------------------------------------------
# PAGE: Configuration — now router only (no cascade)
# -------------------------------------------------------------
def show_configuration():
    """Configuration view — read-only router config from API."""
    st.header("⚙️ System Configuration — Router (Read-only)")

    router_cfg = get_router_info()
    if not router_cfg:
        st.error(
            "Cannot load router configuration. "
            "Ensure API is running and green_ai_config.json exists "
            "under data/models/cascade/."
        )
        return

    # Temperature Scaling
    st.subheader("🔬 Calibration (Temperature Scaling)")

    col1, col2, col3 = st.columns([1, 1, 2])

    T_lr = router_cfg.get("temperature_lr", None)
    T_med = router_cfg.get("temperature_med", None)

    with col1:
        st.metric("LR Temperature", f"{T_lr:.3f}" if T_lr else "N/A")

    with col2:
        st.metric("Medium Temperature", f"{T_med:.3f}" if T_med else "N/A")

    with col3:
        classes = router_cfg.get("classes", Config.CLASSES)
        st.write("**Classes:**")
        st.write(", ".join(classes))

    st.markdown("---")

    # Per-class thresholds
    st.subheader("🎯 Per-class Thresholds (Calibrated)")

    thresholds_lr = router_cfg.get("thresholds_lr", {})
    thresholds_med = router_cfg.get("thresholds_med", {})
    classes = router_cfg.get("classes", Config.CLASSES)

    rows = []
    for idx, cls in enumerate(classes):
        key = str(idx)

        lr_t = thresholds_lr.get(key, thresholds_lr.get(idx, None))
        med_t = thresholds_med.get(key, thresholds_med.get(idx, None))

        rows.append(
            {
                "class": cls,
                "index": idx,
                "lr_threshold": lr_t,
                "med_threshold": med_t,
            }
        )

    df_thresh = pd.DataFrame(rows)
    df_thresh["lr_threshold"] = df_thresh["lr_threshold"].apply(
        lambda x: f"{x:.2f}" if x is not None else "N/A"
    )
    df_thresh["med_threshold"] = df_thresh["med_threshold"].apply(
        lambda x: f"{x:.2f}" if x is not None else "N/A"
    )

    st.table(
        df_thresh[
            ["index", "class", "lr_threshold", "med_threshold"]
        ].rename(
            columns={
                "index": "id",
                "class": "Class",
                "lr_threshold": "LR Threshold",
                "med_threshold": "Medium Threshold",
            }
        )
    )

    st.markdown("---")

    # Validation Stats
    val_stats = router_cfg.get("validation_stats")
    if val_stats:
        st.subheader("📈 Validation Routing Stats")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric(
            "Accuracy",
            f"{val_stats.get('accuracy', 0) * 100:.2f}%",
        )
        c2.metric("LR used", f"{val_stats.get('lr_count', 0)}")
        c3.metric("Medium used", f"{val_stats.get('med_count', 0)}")
        c4.metric("Heavy used", f"{val_stats.get('heavy_count', 0)}")

    st.markdown(
        (
            "ℹ️ **Read-only:** Thresholds are computed offline "
            "(temperature scaling + per-class thresholds). "
            "To update thresholds, run the offline `fit_green_router(...)` "
            "and save `green_ai_config.json` to `data/models/cascade/`."
        )
    )


# -------------------------------------------------------------
# Run the app
# -------------------------------------------------------------
if __name__ == "__main__":
    main()
