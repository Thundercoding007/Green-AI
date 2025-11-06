# File: src/dashboard/app.py
# Streamlit Dashboard for GreenAI Email Classifier

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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
    from src.database import SessionLocal, InferenceLog, get_model_statistics, calculate_energy_savings
    DB_AVAILABLE = True
except Exception:
    DB_AVAILABLE = False

# Page config
st.set_page_config(
    page_title="🌿 GreenAI Email Classifier",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #2ecc71;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #2ecc71;
    }
    .stMetric {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# API Configuration
API_BASE_URL = os.getenv("API_BASE_URL", f"http://{Config.API_HOST}:{Config.API_PORT}")

# Fix for local development: replace 0.0.0.0 with localhost
if "0.0.0.0" in API_BASE_URL:
    API_BASE_URL = API_BASE_URL.replace("0.0.0.0", "localhost")


# Helper functions
@st.cache_data(ttl=60)
def get_api_health():
    """Check API health"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.json()
    except:
        return None

@st.cache_data(ttl=30)
def get_api_stats():
    """Get API statistics"""
    try:
        response = requests.get(f"{API_BASE_URL}/stats", timeout=5)
        return response.json()
    except:
        return None

@st.cache_data(ttl=60)
def get_api_config():
    """Get API configuration"""
    try:
        response = requests.get(f"{API_BASE_URL}/config", timeout=5)
        return response.json()
    except:
        return None

def load_database_stats():
    """Load statistics from database"""
    if not DB_AVAILABLE:
        st.warning("⚠️ Database not available. Some analytics features are disabled.")
        return None

    db = SessionLocal()
    try:
        # Limit to recent logs for performance
        logs = db.query(InferenceLog).order_by(InferenceLog.timestamp.desc()).limit(2000).all()
        if not logs:
            return None

        df = pd.DataFrame([{
            'timestamp': log.timestamp,
            'model_used': log.model_used,
            'prediction': log.predicted_class,
            'confidence': log.confidence,
            'energy_kwh': log.energy_kwh,
            'co2_grams': log.co2_grams,
            'inference_time_ms': log.inference_time_ms,
            'correct': log.correct,
            'cascade_path': log.cascade_path
        } for log in logs])

        return df
    finally:
        db.close()

def classify_email(text: str):
    """Classify email via API"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/classify",
            json={"text": text, "track_energy": True},
            timeout=30
        )
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": response.json().get("detail", "Unknown error")}
    except Exception as e:
        return {"error": str(e)}

def format_energy_kwh(kwh: float):
    """Format energy value for display"""
    if kwh < 0.001:
        return f"{kwh * 1_000_000:.2f} µWh"
    elif kwh < 1:
        return f"{kwh * 1_000:.4f} Wh"
    else:
        return f"{kwh:.6f} kWh"

# Main dashboard
def main():
    # Header
    st.markdown('<p class="main-header">🌿 GreenAI Email Classifier</p>', unsafe_allow_html=True)
    st.markdown("### Sustainable AI-Powered Email Classification")
    
    # Sidebar
    with st.sidebar:
        st.image("https://via.placeholder.com/150x150/2ecc71/ffffff?text=GreenAI", use_column_width=True)
        st.markdown("---")
        
        # API Health Check
        st.subheader("🏥 System Health")
        health = get_api_health()
        
        if health:
            status_color = "🟢" if health['status'] == 'healthy' else "🟡"
            st.markdown(f"{status_color} **Status:** {health['status'].upper()}")
            st.markdown(f"✅ Models Loaded: {'Yes' if health['models_loaded'] else 'No'}")
            st.markdown(f"✅ Cascade Ready: {'Yes' if health['cascade_ready'] else 'No'}")
            st.markdown(f"✅ Database: {'Yes' if health['database_connected'] else 'No'}")
        else:
            st.markdown("🔴 **Status:** API Offline")
            st.error("Cannot connect to API. Please start the API server.")
        
        st.markdown("---")
        
        # Navigation
        st.subheader("📍 Navigation")
        page = st.radio(
            "Select Page",
            ["🏠 Dashboard", "🎯 Classify Email", "📊 Analytics", "⚙️ Configuration"],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        
        # Quick Stats
        stats = get_api_stats()
        if stats:
            st.subheader("📈 Quick Stats")
            st.metric("Total Inferences", f"{stats.get('total_inferences', 0):,}")
            if stats.get('energy_saved_percent'):
                st.metric("Energy Saved", f"{stats['energy_saved_percent']:.1f}%")
    
    # Page routing
    if page == "🏠 Dashboard":
        show_dashboard()
    elif page == "🎯 Classify Email":
        show_classifier()
    elif page == "📊 Analytics":
        show_analytics()
    elif page == "⚙️ Configuration":
        show_configuration()

def show_dashboard():
    """Main dashboard view"""
    st.header("📊 System Overview")
    
    df = load_database_stats()
    stats = get_api_stats()
    
    if df is None or len(df) == 0:
        st.info("📭 No inference data yet. Start classifying emails to see statistics!")
        return
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Inferences",
            f"{len(df):,}",
            delta=f"+{len(df[df['timestamp'] > datetime.now() - timedelta(hours=24)])} today"
        )
    
    with col2:
        avg_confidence = df['confidence'].mean() * 100
        st.metric("Avg Confidence", f"{avg_confidence:.1f}%")
    
    with col3:
        if stats and stats.get('energy_saved_percent'):
            st.metric("Energy Saved", f"{stats['energy_saved_percent']:.1f}%", delta="vs baseline")
        else:
            st.metric("Energy Saved", "N/A")
    
    with col4:
        total_co2 = df['co2_grams'].sum()
        st.metric("CO₂ Emissions", f"{total_co2:.2f}g")
    
    st.markdown("---")

    # Model usage
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🎯 Model Usage Distribution")
        model_counts = df['model_used'].value_counts()
        fig = px.pie(
            values=model_counts.values,
            names=model_counts.index,
            color=model_counts.index,
            color_discrete_map={'green': '#2ecc71', 'medium': '#3498db', 'heavy': '#e74c3c'},
            hole=0.4
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("📈 Classification Distribution")
        class_counts = df['prediction'].value_counts()
        fig = px.bar(
            x=class_counts.index,
            y=class_counts.values,
            color=class_counts.index,
            labels={'x': 'Class', 'y': 'Count'}
        )
        st.plotly_chart(fig, use_container_width=True)

    # Energy/time charts
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("⚡ Energy Consumption Over Time")
        df['hour'] = pd.to_datetime(df['timestamp']).dt.floor('H')
        hourly = df.groupby('hour')['energy_kwh'].sum().reset_index()
        fig = px.line(hourly, x='hour', y='energy_kwh', labels={'hour': 'Time', 'energy_kwh': 'Energy (kWh)'})
        fig.update_traces(line_color='#2ecc71', line_width=3)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.subheader("🚀 Inference Time by Model")
        fig = px.box(
            df, x='model_used', y='inference_time_ms', color='model_used',
            color_discrete_map={'green': '#2ecc71', 'medium': '#3498db', 'heavy': '#e74c3c'},
        )
        st.plotly_chart(fig, use_container_width=True)

    # Recent inferences
    st.subheader("📝 Recent Inferences")
    recent_df = df.head(10)[['timestamp', 'prediction', 'confidence', 'model_used', 'inference_time_ms', 'energy_kwh']]
    recent_df['confidence'] = recent_df['confidence'].apply(lambda x: f"{x*100:.1f}%")
    recent_df['inference_time_ms'] = recent_df['inference_time_ms'].apply(lambda x: f"{x:.2f}ms")
    recent_df['energy_kwh'] = recent_df['energy_kwh'].apply(format_energy_kwh)
    st.dataframe(recent_df, use_container_width=True)

def show_classifier():
    """Email classification interface"""
    st.header("🎯 Email Classification")
    
    health = get_api_health()
    if not health or health['status'] != 'healthy':
        st.error("⚠️ API is not available. Please start the API server first.")
        return
    
    tab1, tab2 = st.tabs(["📝 Single Email", "📚 Batch Upload"])
    
    with tab1:
        st.subheader("Classify a Single Email")
        samples = {
            "Meeting Request": "Hi team, let's schedule a meeting tomorrow at 2pm to discuss the quarterly results.",
            "Promotional": "HUGE SALE! Get 70% off on all products. Limited time offer. Shop now!",
            "Support Ticket": "Your support ticket #12345 has been received. Our team will respond within 24 hours.",
            "Personal": "Hey! Want to grab coffee this weekend? It's been too long since we caught up."
        }
        selected_sample = st.selectbox("Try a sample email:", ["Custom"] + list(samples.keys()))
        if st.button("Load Sample") and selected_sample != "Custom":
            default_text = samples[selected_sample]
        else:
            default_text = ""
        email_text = st.text_area("Email Text:", value=default_text, height=200)
        if st.button("🚀 Classify", type="primary"):
            if len(email_text) < 10:
                st.error("Please enter at least 10 characters")
            else:
                with st.spinner("🔄 Classifying..."):
                    result = classify_email(email_text)
                if "error" in result:
                    st.error(f"❌ Error: {result['error']}")
                else:
                    st.success("✅ Classification Complete!")
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Prediction", result['prediction'].upper())
                    col2.metric("Confidence", f"{result['confidence']*100:.1f}%")
                    col3.metric("Model Used", result['model_used'].upper())
                    col4.metric("Time", f"{result['inference_time_ms']:.2f}ms")
                    with st.expander("📊 Detailed Results"):
                        st.write(f"- Cascade Path: `{result['cascade_path']}`")
                        if result.get('energy_kwh'):
                            st.write(f"- Energy: {format_energy_kwh(result['energy_kwh'])}")
                        if result.get('co2_grams'):
                            st.write(f"- CO₂: {result['co2_grams']:.6f} g")
                    fig = px.bar(
                        x=list(result['probabilities'].keys()),
                        y=list(result['probabilities'].values()),
                        color=list(result['probabilities'].keys())
                    )
                    st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("Batch Classification")
        uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file)
                if 'text' not in df.columns:
                    st.error("CSV must contain a 'text' column")
                    return
                st.write(f"Loaded {len(df)} emails")
                if st.button("🚀 Classify All", type="primary"):
                    progress = st.progress(0)
                    results = []
                    for i, text in enumerate(df['text']):
                        result = classify_email(str(text))
                        results.append(result)
                        progress.progress((i+1)/len(df))
                    df['prediction'] = [r.get('prediction', 'error') for r in results]
                    df['confidence'] = [r.get('confidence', 0) for r in results]
                    df['model_used'] = [r.get('model_used', 'error') for r in results]
                    st.success("✅ Batch classification complete!")
                    st.dataframe(df, use_container_width=True)
                    st.download_button("📥 Download Results", df.to_csv(index=False), "classification_results.csv", "text/csv")
            except Exception as e:
                st.error(f"Error processing file: {e}")

def show_analytics():
    """Advanced analytics view"""
    st.header("📊 Advanced Analytics")
    df = load_database_stats()
    if df is None or len(df) == 0:
        st.info("📭 No data available for analytics yet.")
        return
    date_range = st.date_input("Date Range", value=(df['timestamp'].min().date(), df['timestamp'].max().date()))
    model_filter = st.multiselect("Models", options=['green', 'medium', 'heavy'], default=['green', 'medium', 'heavy'])
    class_filter = st.multiselect("Classes", options=df['prediction'].unique().tolist(), default=df['prediction'].unique().tolist())
    filtered_df = df[(df['model_used'].isin(model_filter)) & (df['prediction'].isin(class_filter))]
    st.metric("Filtered Records", f"{len(filtered_df):,}")
    tab1, _, _ = st.tabs(["📈 Trends", "🎯 Performance", "⚡ Efficiency"])
    with tab1:
        filtered_df['date'] = pd.to_datetime(filtered_df['timestamp']).dt.date
        daily = filtered_df.groupby('date').size().reset_index(name='count')
        fig = px.line(daily, x='date', y='count', labels={'date': 'Date', 'count': 'Inferences'})
        fig.update_traces(line_color='#2ecc71', line_width=3)
        st.plotly_chart(fig, use_container_width=True)

def show_configuration():
    """Configuration view"""
    st.header("⚙️ System Configuration")
    config = get_api_config()
    if not config:
        st.error("Cannot load configuration from API")
        return
    st.subheader("🎚️ Cascade Thresholds")
    col1, col2 = st.columns(2)
    green_threshold = st.slider("Green Model Threshold", 0.5, 1.0, config['green_threshold'], 0.05)
    medium_threshold = st.slider("Medium Model Threshold", 0.5, min(green_threshold, 1.0), config['medium_threshold'], 0.05)
    if st.button("💾 Update Thresholds", type="primary"):
        try:
            response = requests.post(
                f"{API_BASE_URL}/config/thresholds",
                json={
                    'green_threshold': green_threshold,
                    'medium_threshold': medium_threshold
                },
                timeout=5
            )
            if response.status_code == 200:
                st.success("✅ Thresholds updated successfully!")
                st.rerun()
            else:
                st.error(f"❌ Error: {response.json().get('detail', 'Unknown error')}")
        except Exception as e:
            st.error(f"❌ Error: {e}")

    st.markdown("---")
    st.subheader("🤖 Model Information")
    if 'model_info' in config:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("### 🌱 Green Model")
            st.write(f"**Type:** {config['model_info']['green']['type']}")
            st.write(f"**Size:** {config['model_info']['green']['size_mb']:.2f} MB")
        with col2:
            st.markdown("### 🔬 Medium Model")
            st.write(f"**Type:** {config['model_info']['medium']['type']}")
            st.write(f"**Size:** {config['model_info']['medium']['size_mb']:.2f} MB")
        with col3:
            st.markdown("### 🚀 Heavy Model")
            st.write(f"**Type:** {config['model_info']['heavy']['type']}")
            st.write(f"**Size:** {config['model_info']['heavy']['size_mb']:.2f} MB")

    st.markdown("---")
    st.subheader("ℹ️ System Information")
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Classes:**")
        for cls in config.get('classes', []):
            st.write(f"- {cls}")
    with col2:
        st.write("**Current Thresholds:**")
        st.write(f"- Green: {config['green_threshold']:.2f}")
        st.write(f"- Medium: {config['medium_threshold']:.2f}")

# Run the app
if __name__ == "__main__":
    main()
