"""
Streamlit Frontend: RetentionAI: Predictive Customer Intelligence Platform
Executive Design System: Luminous Dark Glassmorphism, Fresh Vibrant Accents, Instinctive Interactive Layout
Tab 1: 🎯 Retention Intelligence & Customer Profiler (Unified Macro Survival + Micro Interactive Profiler)
Tab 2: 📁 Enterprise Batch Prediction (Up to 1M+ rows & 1,000+ cols streaming)
Tab 3: 🗄️ Analytical Database & SQL Workbench (DuckDB Columnar OLAP)
"""

import os
import json
import time
import urllib.request
import urllib.error
import joblib
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

# --------------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------------
st.set_page_config(
    page_title="RetentionAI: Predictive Customer Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------------
# ENTERPRISE SAAS DESIGN SYSTEM (SLATE PALETTE, CRISP HIERARCHY, ACCESSIBLE CONTRAST)
# --------------------------------------------------------
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Outfit:wght@400;500;600;700;800&family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">

<style>
    /* ----------------------------------------------------
       GLOBAL BASE & ENTERPRISE PALETTE (#0F172A DEEP SLATE)
       ---------------------------------------------------- */
    html, body, [class*="css"], .stApp {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
        background-color: #0F172A !important;
        color: #F8FAFC !important;
    }

    /* Main App Background Override */
    .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
        background-color: #0F172A !important;
        color: #F8FAFC !important;
    }
    
    [data-testid="stToolbar"] {
        background-color: #0F172A !important;
    }

    /* ----------------------------------------------------
       TYPOGRAPHY STANDARDS: #F8FAFC HEADERS & #94A3B8 MUTED TEXT
       ---------------------------------------------------- */
    h1, h2, h3, h4, h5, h6, .hero-title {
        font-family: 'Outfit', 'Plus Jakarta Sans', sans-serif !important;
        color: #F8FAFC !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em;
    }

    p, span, label, .hero-subtitle, .kpi-label, .kpi-sub, caption, [data-testid="stMarkdownContainer"] p, div[data-testid="stCaptionContainer"] {
        color: #94A3B8 !important;
        font-weight: 400;
    }

    strong, b {
        color: #F8FAFC !important;
        font-weight: 600;
    }

    /* ----------------------------------------------------
       CONTAINERS, METRIC CARDS & INPUT BOXES
       Background: #1E293B | Border: 1px solid #334155 | Border-Radius: 8px
       Drop Shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3), 0 2px 4px -1px rgba(0, 0, 0, 0.18)
       ---------------------------------------------------- */
    .saas-card,
    .kpi-card,
    .rec-card,
    .gemini-card,
    div[data-testid="stMetric"],
    [data-testid="stForm"],
    [data-testid="stExpander"] {
        background-color: #1E293B !important;
        border: 1px solid #334155 !important;
        border-radius: 8px !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3), 0 2px 4px -1px rgba(0, 0, 0, 0.18) !important;
    }

    /* Metric Cards Specifics */
    div[data-testid="stMetric"] {
        padding: 1.15rem 1.25rem !important;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    div[data-testid="stMetric"]:hover, .kpi-card:hover {
        border-color: #475569 !important;
        transform: translateY(-2px);
    }
    [data-testid="stMetricLabel"] p {
        color: #94A3B8 !important;
        font-size: 0.82rem !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.05em !important;
    }
    [data-testid="stMetricValue"] div {
        color: #F8FAFC !important;
        font-family: 'Outfit', sans-serif !important;
        font-weight: 800 !important;
        font-size: 1.85rem !important;
    }

    /* KPI Cards in Ribbon */
    .kpi-card {
        padding: 1.25rem 1.4rem;
        position: relative;
        overflow: hidden;
    }
    .kpi-card-cyan { border-top: 3px solid #38BDF8 !important; }
    .kpi-card-mint { border-top: 3px solid #10B981 !important; }
    .kpi-card-indigo { border-top: 3px solid #818CF8 !important; }
    .kpi-card-coral { border-top: 3px solid #F43F5E !important; }

    .kpi-label {
        font-size: 0.8rem;
        text-transform: uppercase;
        font-weight: 700;
        letter-spacing: 0.05em;
        margin-bottom: 0.35rem;
        color: #94A3B8 !important;
    }
    .kpi-value {
        font-family: 'Outfit', sans-serif;
        font-size: 2.1rem;
        font-weight: 800;
        color: #F8FAFC !important;
        line-height: 1.1;
    }
    .kpi-sub {
        font-size: 0.82rem;
        margin-top: 0.35rem;
        font-weight: 500;
        color: #94A3B8 !important;
    }

    /* Hero Header */
    .hero-container {
        background-color: #1E293B;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 1.75rem 2.25rem;
        margin-bottom: 1.75rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3), 0 2px 4px -1px rgba(0, 0, 0, 0.18);
        position: relative;
    }
    .hero-container::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
        background: #3B82F6;
        border-radius: 8px 8px 0 0;
    }
    .hero-title {
        font-size: 2.25rem;
        font-weight: 800;
        color: #F8FAFC !important;
        letter-spacing: -0.025em;
        margin-bottom: 0.3rem;
    }
    .hero-subtitle {
        color: #94A3B8 !important;
        font-size: 1rem;
        line-height: 1.5;
    }

    /* ----------------------------------------------------
       INPUT BOXES & CONTROLS (#1E293B, #334155, 8px RADIUS)
       ---------------------------------------------------- */
    div[data-baseweb="input"],
    div[data-baseweb="select"] > div,
    div[data-testid="stNumberInput"] div[data-baseweb="input"],
    div[data-baseweb="textarea"] textarea,
    [data-testid="stFileUploadDropzone"],
    .stTextInput input,
    .stSelectbox select {
        background-color: #1E293B !important;
        border: 1px solid #334155 !important;
        border-radius: 8px !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3), 0 2px 4px -1px rgba(0, 0, 0, 0.18) !important;
        color: #F8FAFC !important;
    }

    div[data-baseweb="input"] input {
        color: #F8FAFC !important;
        background-color: transparent !important;
    }

    div[data-baseweb="select"] span {
        color: #F8FAFC !important;
    }

    /* Dropdown Menus */
    ul[data-baseweb="menu"] {
        background-color: #1E293B !important;
        border: 1px solid #334155 !important;
        border-radius: 8px !important;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.5) !important;
    }
    li[data-baseweb="menu-item"] {
        color: #CBD5E1 !important;
    }
    li[data-baseweb="menu-item"]:hover {
        background-color: #334155 !important;
        color: #F8FAFC !important;
    }

    /* Expander Details */
    [data-testid="stExpander"] {
        border-radius: 8px !important;
    }
    [data-testid="stExpander"] details {
        border-radius: 8px !important;
    }
    [data-testid="stExpander"] summary {
        color: #F8FAFC !important;
        font-weight: 600 !important;
    }

    /* ----------------------------------------------------
       BUTTONS OVERRIDE: SOLID ROYAL BLUE #3B82F6 (HOVER BRIGHTNESS)
       ---------------------------------------------------- */
    div.stButton > button,
    div[data-testid="stFormSubmitButton"] > button,
    div[data-testid="stDownloadButton"] > button,
    button[kind="primary"],
    button[kind="secondary"] {
        background-color: #3B82F6 !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        padding: 0.6rem 1.25rem !important;
        transition: filter 0.2s ease, transform 0.15s ease, box-shadow 0.2s ease !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3), 0 2px 4px -1px rgba(0, 0, 0, 0.18) !important;
    }

    div.stButton > button:hover,
    div[data-testid="stFormSubmitButton"] > button:hover,
    div[data-testid="stDownloadButton"] > button:hover,
    button[kind="primary"]:hover,
    button[kind="secondary"]:hover {
        filter: brightness(1.1) !important;
        background-color: #3B82F6 !important;
        color: #FFFFFF !important;
        border: none !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 12px -2px rgba(59, 130, 246, 0.35) !important;
    }

    div.stButton > button:active,
    div[data-testid="stFormSubmitButton"] > button:active,
    div[data-testid="stDownloadButton"] > button:active {
        filter: brightness(0.95) !important;
        transform: translateY(0) !important;
    }

    /* ----------------------------------------------------
       TABS STYLING (SEGMENTED ENTERPRISE CONTROLS)
       ---------------------------------------------------- */
    button[data-baseweb="tab"] {
        font-family: 'Outfit', sans-serif !important;
        font-size: 1rem !important;
        font-weight: 600 !important;
        color: #94A3B8 !important;
        padding: 10px 22px !important;
        border-radius: 8px 8px 0 0 !important;
        border-bottom: 2px solid transparent !important;
        transition: all 0.2s ease !important;
    }
    button[data-baseweb="tab"]:hover {
        color: #CBD5E1 !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #F8FAFC !important;
        border-bottom: 2px solid #3B82F6 !important;
        background-color: rgba(59, 130, 246, 0.08) !important;
    }

    /* Badge Pills */
    .badge-pill {
        display: inline-flex;
        align-items: center;
        padding: 4px 12px;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 700;
        letter-spacing: 0.04em;
        text-transform: uppercase;
    }
    .badge-danger {
        background-color: rgba(239, 68, 68, 0.15);
        color: #FCA5A5;
        border: 1px solid rgba(239, 68, 68, 0.4);
    }
    .badge-warning {
        background-color: rgba(245, 158, 11, 0.15);
        color: #FCD34D;
        border: 1px solid rgba(245, 158, 11, 0.4);
    }
    .badge-success {
        background-color: rgba(16, 185, 129, 0.15);
        color: #6EE7B7;
        border: 1px solid rgba(16, 185, 129, 0.4);
    }
    .badge-info {
        background-color: rgba(59, 130, 246, 0.15);
        color: #93C5FD;
        border: 1px solid rgba(59, 130, 246, 0.4);
    }

    /* Prescriptive Action Recommendations Playbook */
    .rec-card-high {
        background-color: #1E293B;
        border: 1px solid #334155;
        border-left: 4px solid #EF4444;
        border-radius: 8px;
        padding: 1.25rem 1.4rem;
        margin-top: 1rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3), 0 2px 4px -1px rgba(0, 0, 0, 0.18);
    }
    .rec-card-low {
        background-color: #1E293B;
        border: 1px solid #334155;
        border-left: 4px solid #10B981;
        border-radius: 8px;
        padding: 1.25rem 1.4rem;
        margin-top: 1rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3), 0 2px 4px -1px rgba(0, 0, 0, 0.18);
    }

    /* Dataframe Table Dark Theme */
    [data-testid="stDataFrame"] {
        background-color: #1E293B !important;
        border: 1px solid #334155 !important;
        border-radius: 8px !important;
        overflow: hidden;
    }
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------------
# CONSTANTS & CACHED LOADERS
# --------------------------------------------------------
BACKEND_URL = os.environ.get("BACKEND_URL", "http://127.0.0.1:8000")
ARTIFACTS_DIR = "artifacts"
KM_DATA_PATH = os.path.join(ARTIFACTS_DIR, "km_data.joblib")

@st.cache_data
def load_km_data():
    if os.path.exists(KM_DATA_PATH):
        return joblib.load(KM_DATA_PATH)
    try:
        req = urllib.request.Request(f"{BACKEND_URL}/survival-macro", headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=3) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception:
        return None

def call_predict_churn_api(payload: dict) -> dict:
    url = f"{BACKEND_URL}/predict-churn"
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=5) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as e:
        if os.path.exists(os.path.join(ARTIFACTS_DIR, "xgb_model.joblib")):
            return local_predict_fallback(payload)
        raise RuntimeError(f"Failed to connect to backend at {url}. Error: {e}")

def local_predict_fallback(payload: dict) -> dict:
    import shap
    preprocessor = joblib.load(os.path.join(ARTIFACTS_DIR, "preprocessor.joblib"))
    xgb_model = joblib.load(os.path.join(ARTIFACTS_DIR, "xgb_model.joblib"))
    metadata = joblib.load(os.path.join(ARTIFACTS_DIR, "metadata.joblib"))
    feature_names = metadata["transformed_feature_names"]

    df = pd.DataFrame([payload])
    X_trans = preprocessor.transform(df)
    prob = float(xgb_model.predict_proba(X_trans)[0, 1])
    
    explainer = shap.TreeExplainer(xgb_model)
    shap_vals = explainer.shap_values(X_trans)
    if isinstance(shap_vals, list):
        sample_shap = np.array(shap_vals[1][0], dtype=float)
    else:
        sample_shap = np.array(shap_vals[0], dtype=float)
    
    base_val = explainer.expected_value
    if isinstance(base_val, (list, np.ndarray)):
        base_val = float(base_val[-1])
    else:
        base_val = float(base_val)

    contributions = []
    for f_name, s_val in zip(feature_names, sample_shap):
        impact = "Increases Churn Risk" if s_val > 0 else "Decreases Churn Risk (Protective)"
        contributions.append({
            "feature": f_name,
            "contribution": round(float(s_val), 4),
            "abs_contribution": round(float(abs(s_val)), 4),
            "description": f"{impact} by {abs(s_val):.3f}"
        })
    retention_strategy = None
    gemini_key = (payload.get("gemini_api_key") or os.environ.get("GEMINI_API_KEY", "")).strip()
    if gemini_key:
        try:
            import google.generativeai as genai
            genai.configure(api_key=gemini_key)
            gemini_model = genai.GenerativeModel("gemini-1.5-flash")
            top_3 = contributions[:3]
            top_3_text = "\n".join([f"- **{d['feature']}**: {d['description']}" for d in top_3])
            prompt = f"""You are RetentionAI, an expert customer retention strategist.
Customer Churn Probability: {prob:.1%} ({tier})
Top 3 Highest-Contributing SHAP Risk Drivers:
{top_3_text}
Please provide a structured, personalized customer retention strategy with:
1. Root-Cause Diagnostic
2. Recommended Retention Action
3. Specific Discount & Incentive Offer
4. Long-Term Value Plan
"""
            resp = gemini_model.generate_content(prompt)
            if resp and resp.text:
                retention_strategy = resp.text.strip()
        except Exception as e:
            retention_strategy = f"⚠️ Gemini API Notice: {str(e)}"

    tier = "High Risk" if prob >= 0.65 else ("Moderate Risk" if prob >= 0.35 else "Low Risk")
    return {
        "churn_probability": round(prob, 4),
        "churn_prediction": int(prob >= 0.5),
        "risk_tier": tier,
        "base_value": round(base_val, 4),
        "total_shap_margin": round(float(np.sum(sample_shap) + base_val), 4),
        "feature_contributions": contributions,
        "raw_shap_values": [round(float(v), 5) for v in sample_shap],
        "feature_names": list(feature_names),
        "retention_strategy": retention_strategy,
        "input_summary": payload
    }

def call_predict_batch_api(file_bytes: bytes, filename: str, output_format: str = "auto") -> tuple:
    import requests
    from io import BytesIO
    import pyarrow.parquet as pq
    from batch_processor import detect_file_format

    in_fmt = detect_file_format(filename)
    target_fmt = in_fmt if output_format == "auto" else output_format
    
    url = f"{BACKEND_URL}/predict-batch?output_format={target_fmt}&return_mode=file"
    files = {"file": (filename, file_bytes, "application/octet-stream")}
    
    try:
        resp = requests.post(url, files=files, timeout=300)
        if resp.status_code != 200:
            raise RuntimeError(f"Backend error ({resp.status_code}): {resp.text}")
        
        scored_bytes = resp.content
        total_rows = int(resp.headers.get("X-Total-Rows", 0))
        total_cols = int(resp.headers.get("X-Total-Columns", 0))
        mean_risk = float(resp.headers.get("X-Mean-Risk-Score", 0.0))
        high_risk_count = int(resp.headers.get("X-High-Risk-Count", 0))
        low_risk_count = int(resp.headers.get("X-Low-Risk-Count", max(0, total_rows - high_risk_count)))
        moderate_risk_count = int(resp.headers.get("X-Moderate-Risk-Count", 0))

        preview_df = None
        if target_fmt == "parquet":
            pq_file = pq.ParquetFile(BytesIO(scored_bytes))
            preview_batch = next(pq_file.iter_batches(batch_size=500))
            preview_df = preview_batch.to_pandas()
            if total_rows == 0:
                total_rows = pq_file.metadata.num_rows
                total_cols = pq_file.metadata.num_columns
        elif target_fmt == "csv":
            preview_df = pd.read_csv(BytesIO(scored_bytes), nrows=500)
            if total_cols == 0:
                total_cols = preview_df.shape[1]
        elif target_fmt in ["jsonl", "json"]:
            try:
                preview_df = pd.read_json(BytesIO(scored_bytes), lines=True, nrows=500)
            except Exception:
                preview_df = pd.read_json(BytesIO(scored_bytes)).head(500)
            if total_cols == 0 and preview_df is not None:
                total_cols = preview_df.shape[1]
        else:
            preview_df = pd.read_csv(BytesIO(scored_bytes), nrows=500)

        stats = {
            "total_rows": total_rows if total_rows > 0 else (len(preview_df) if preview_df is not None else 0),
            "total_columns": total_cols if total_cols > 0 else (preview_df.shape[1] if preview_df is not None else 0),
            "mean_risk_score": mean_risk,
            "high_risk_count": high_risk_count,
            "low_risk_count": low_risk_count,
            "moderate_risk_count": moderate_risk_count,
            "output_format": target_fmt
        }
        return scored_bytes, preview_df, stats, target_fmt

    except Exception as e:
        if os.path.exists(os.path.join(ARTIFACTS_DIR, "xgb_model.joblib")):
            return local_batch_large_fallback(file_bytes, filename, target_fmt)
        raise e

def local_batch_large_fallback(file_bytes: bytes, filename: str, target_fmt: str) -> tuple:
    import tempfile
    from batch_processor import process_large_dataset, detect_file_format
    
    in_fmt = detect_file_format(filename)
    temp_dir = tempfile.gettempdir()
    in_temp = os.path.join(temp_dir, f"in_{os.urandom(6).hex()}_{filename}")
    out_temp = os.path.join(temp_dir, f"out_{os.urandom(6).hex()}.{target_fmt}")
    
    with open(in_temp, "wb") as f:
        f.write(file_bytes)
        
    preprocessor = joblib.load(os.path.join(ARTIFACTS_DIR, "preprocessor.joblib"))
    xgb_model = joblib.load(os.path.join(ARTIFACTS_DIR, "xgb_model.joblib"))
    
    stats = process_large_dataset(in_temp, out_temp, in_fmt, target_fmt, preprocessor, xgb_model)
    with open(out_temp, "rb") as f:
        scored_bytes = f.read()
        
    try:
        os.remove(in_temp)
        os.remove(out_temp)
    except Exception:
        pass
        
    preview_df = pd.DataFrame(stats["preview_records"])
    return scored_bytes, preview_df, stats, target_fmt

# --------------------------------------------------------
# EXECUTIVE HERO HEADER
# --------------------------------------------------------
st.markdown("""
<div class="hero-container">
    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
        <div>
            <div class="hero-title">⚡ RetentionAI: Predictive Customer Intelligence</div>
            <div class="hero-subtitle">Unified Micro Customer Risk Profiler & Macro Cohort Survival Dynamics • Real-Time XGBoost, Tree SHAP & DuckDB OLAP</div>
        </div>
        <div style="text-align: right; margin-top: 10px;">
            <span class="badge-pill badge-success" style="font-size: 0.88rem; box-shadow: 0 0 15px rgba(16, 185, 129, 0.4);">🟢 System Operational</span>
            <div style="color: #38BDF8; font-size: 0.82rem; margin-top: 5px; font-weight: 600;">⚡ Engine Latency: &lt; 8ms</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# --------------------------------------------------------
# SIDEBAR
# --------------------------------------------------------
with st.sidebar:
    st.markdown("### 🎛️ Platform Controls")
    
    # Engine status pill
    backend_status = "🔴 Disconnected"
    try:
        req = urllib.request.Request(f"{BACKEND_URL}/health", timeout=1)
        with urllib.request.urlopen(req) as resp:
            if resp.status == 200:
                backend_status = "🟢 Live (FastAPI 8000)"
    except Exception:
        if os.path.exists(os.path.join(ARTIFACTS_DIR, "xgb_model.joblib")):
            backend_status = "🟡 Embedded Joblib"
            
    st.markdown(f"""
    <div class="kpi-card kpi-card-cyan" style="padding: 1rem; margin-bottom: 1.2rem;">
        <div class="kpi-label">API Gateway Status</div>
        <div style="font-weight: 700; color: #00F2FE; font-size: 1.05rem;">{backend_status}</div>
        <div style="font-size: 0.75rem; color: #94A3B8; margin-top: 4px;">Host: {BACKEND_URL}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 🧠 Model Intelligence")
    st.markdown("""
    - **Macro Engine:** Lifelines Non-parametric Kaplan-Meier
    - **Micro Engine:** XGBoost Calibrated Classifier
    - **Explainability:** Lundberg Tree SHAP Log-Odds
    - **Database:** Embedded DuckDB Vectorized OLAP
    """)
    st.markdown("---")
    st.caption("v2.5 Enterprise Edition • Production Build")

# --------------------------------------------------------
# NAVIGATION TABS
# --------------------------------------------------------
tab1, tab2, tab3 = st.tabs([
    "🎯 Micro Customer Profiling & Survival Analysis",
    "📁 Enterprise Batch Prediction",
    "🗄️ Analytical Database & SQL Workbench"
])

# ========================================================
# TAB 1: ALL USER FEATURES + MACRO SURVIVAL (UNIFIED 1ST PAGE)
# ========================================================
with tab1:
    # Top KPI Pulse Ribbon (Instant Instinctive Overview)
    km_data = load_km_data()
    median_st = km_data["overall"].get("median_survival_time") if km_data else 64.0
    timeline = np.array(km_data["overall"]["timeline"]) if km_data else np.array([0, 12, 24, 72])
    survival_prob = np.array(km_data["overall"]["survival_prob"]) if km_data else np.array([1, 0.88, 0.76, 0.65])
    
    idx_12 = (np.abs(timeline - 12)).argmin()
    idx_24 = (np.abs(timeline - 24)).argmin()
    prob_12 = survival_prob[idx_12] * 100
    prob_24 = survival_prob[idx_24] * 100

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.markdown(f"""
        <div class="kpi-card kpi-card-cyan">
            <div class="kpi-label">Total Cohort Monitored</div>
            <div class="kpi-value" style="color: #38BDF8;">100,000</div>
            <div class="kpi-sub">Active Subscriber Lifelines</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi2:
        st.markdown(f"""
        <div class="kpi-card kpi-card-mint">
            <div class="kpi-label">1-Year Survival Rate</div>
            <div class="kpi-value" style="color: #10B981;">{prob_12:.1f}%</div>
            <div class="kpi-sub" style="color: #34D399;">Retained through M12</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi3:
        st.markdown(f"""
        <div class="kpi-card kpi-card-indigo">
            <div class="kpi-label">2-Year Survival Rate</div>
            <div class="kpi-value" style="color: #818CF8;">{prob_24:.1f}%</div>
            <div class="kpi-sub" style="color: #A5B4FC;">Retained through M24</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi4:
        st.markdown(f"""
        <div class="kpi-card kpi-card-coral">
            <div class="kpi-label">Highest Hazard Cohort</div>
            <div class="kpi-value" style="color: #F43F5E;">39.2%</div>
            <div class="kpi-sub" style="color: #FDA4AF;">Month-to-month + Fiber</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

    # TWO-COLUMN SPLIT: LEFT COMMAND DECK (INPUTS) VS RIGHT REVELATION (OUTPUTS & SHAP)
    deck_col, viz_col = st.columns([1, 1.25], gap="large")

    # ----------------------------------------------------
    # LEFT: CUSTOMER PROFILING DECK
    # ----------------------------------------------------
    with deck_col:
        st.markdown("### 👤 Customer Profile Designer")
        st.caption("Adjust customer attributes or load 1-click behavioral archetypes below.")

        # Preset selection buttons
        st.write("**⚡ Instant Archetype Presets:**")
        p_c1, p_c2, p_c3 = st.columns(3)

        if "form_data" not in st.session_state:
            st.session_state["form_data"] = {
                "tenure": 3,
                "MonthlyCharges": 94.50,
                "TotalCharges": 283.50,
                "gender": "Female",
                "SeniorCitizen": 0,
                "Partner": "No",
                "Dependents": "No",
                "PhoneService": "Yes",
                "MultipleLines": "Yes",
                "InternetService": "Fiber optic",
                "OnlineSecurity": "No",
                "OnlineBackup": "No",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "Yes",
                "StreamingMovies": "Yes",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check"
            }

        with p_c1:
            if st.button("🚨 High Flight Risk", use_container_width=True):
                st.session_state["form_data"] = {
                    "tenure": 2,
                    "MonthlyCharges": 98.75,
                    "TotalCharges": 197.50,
                    "gender": "Female",
                    "SeniorCitizen": 1,
                    "Partner": "No",
                    "Dependents": "No",
                    "PhoneService": "Yes",
                    "MultipleLines": "Yes",
                    "InternetService": "Fiber optic",
                    "OnlineSecurity": "No",
                    "OnlineBackup": "No",
                    "DeviceProtection": "No",
                    "TechSupport": "No",
                    "StreamingTV": "Yes",
                    "StreamingMovies": "Yes",
                    "Contract": "Month-to-month",
                    "PaperlessBilling": "Yes",
                    "PaymentMethod": "Electronic check"
                }
                st.rerun()

        with p_c2:
            if st.button("🛡️ Loyal VIP Cohort", use_container_width=True):
                st.session_state["form_data"] = {
                    "tenure": 60,
                    "MonthlyCharges": 42.10,
                    "TotalCharges": 2526.00,
                    "gender": "Male",
                    "SeniorCitizen": 0,
                    "Partner": "Yes",
                    "Dependents": "Yes",
                    "PhoneService": "Yes",
                    "MultipleLines": "No",
                    "InternetService": "DSL",
                    "OnlineSecurity": "Yes",
                    "OnlineBackup": "Yes",
                    "DeviceProtection": "Yes",
                    "TechSupport": "Yes",
                    "StreamingTV": "No",
                    "StreamingMovies": "No",
                    "Contract": "Two year",
                    "PaperlessBilling": "No",
                    "PaymentMethod": "Bank transfer (automatic)"
                }
                st.rerun()

        with p_c3:
            if st.button("⚖️ Mid-Tenure At-Risk", use_container_width=True):
                st.session_state["form_data"] = {
                    "tenure": 14,
                    "MonthlyCharges": 72.30,
                    "TotalCharges": 1012.20,
                    "gender": "Male",
                    "SeniorCitizen": 0,
                    "Partner": "Yes",
                    "Dependents": "No",
                    "PhoneService": "Yes",
                    "MultipleLines": "No",
                    "InternetService": "Fiber optic",
                    "OnlineSecurity": "No",
                    "OnlineBackup": "Yes",
                    "DeviceProtection": "No",
                    "TechSupport": "No",
                    "StreamingTV": "No",
                    "StreamingMovies": "No",
                    "Contract": "Month-to-month",
                    "PaperlessBilling": "Yes",
                    "PaymentMethod": "Credit card (automatic)"
                }
                st.rerun()

        cur = st.session_state["form_data"]

        # Clean Interactive Form
        with st.form("interactive_customer_form"):
            st.markdown("##### 📌 Subscription & Agreement Terms")
            f_col1, f_col2 = st.columns(2)
            with f_col1:
                contract = st.selectbox(
                    "Contract Duration",
                    ["Month-to-month", "One year", "Two year"],
                    index=["Month-to-month", "One year", "Two year"].index(cur["Contract"])
                )
                tenure = st.slider("Account Tenure (Months)", min_value=1, max_value=72, value=int(cur["tenure"]))
            with f_col2:
                payment_method = st.selectbox(
                    "Payment Method",
                    ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
                    index=["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"].index(cur["PaymentMethod"])
                )
                monthly_charges = st.number_input("Monthly Charges ($)", min_value=15.0, max_value=150.0, value=float(cur["MonthlyCharges"]), step=1.0)

            total_charges = float(np.round(tenure * monthly_charges, 2))

            st.markdown("##### 🌐 Network & Bundled Add-Ons")
            f_col3, f_col4 = st.columns(2)
            with f_col3:
                internet_service = st.selectbox(
                    "Internet Service Tier",
                    ["Fiber optic", "DSL", "No"],
                    index=["Fiber optic", "DSL", "No"].index(cur["InternetService"])
                )
                tech_support = st.selectbox(
                    "Tech Support Plan",
                    ["No", "Yes", "No internet service"],
                    index=["No", "Yes", "No internet service"].index(cur["TechSupport"])
                )
                online_security = st.selectbox(
                    "Online Security Shield",
                    ["No", "Yes", "No internet service"],
                    index=["No", "Yes", "No internet service"].index(cur["OnlineSecurity"])
                )
            with f_col4:
                online_backup = st.selectbox(
                    "Cloud Backup Service",
                    ["No", "Yes", "No internet service"],
                    index=["No", "Yes", "No internet service"].index(cur["OnlineBackup"])
                )
                device_protection = st.selectbox(
                    "Device Protection Plan",
                    ["No", "Yes", "No internet service"],
                    index=["No", "Yes", "No internet service"].index(cur["DeviceProtection"])
                )
                paperless_billing = st.selectbox(
                    "Paperless e-Billing",
                    ["Yes", "No"],
                    index=["Yes", "No"].index(cur["PaperlessBilling"])
                )

            with st.expander("👤 Customer Demographics & Secondary Services"):
                d_c1, d_c2 = st.columns(2)
                with d_c1:
                    gender = st.selectbox("Gender", ["Female", "Male"], index=["Female", "Male"].index(cur["gender"]))
                    senior_citizen = st.selectbox("Senior Citizen (65+)", [0, 1], index=[0, 1].index(cur["SeniorCitizen"]), format_func=lambda x: "Yes" if x == 1 else "No")
                    partner = st.selectbox("Has Partner", ["No", "Yes"], index=["No", "Yes"].index(cur["Partner"]))
                with d_c2:
                    dependents = st.selectbox("Has Dependents", ["No", "Yes"], index=["No", "Yes"].index(cur["Dependents"]))
                    phone_service = st.selectbox("Phone Service", ["Yes", "No"], index=["Yes", "No"].index(cur["PhoneService"]))
                    multiple_lines = st.selectbox("Multiple Phone Lines", ["No", "Yes", "No phone service"], index=["No", "Yes", "No phone service"].index(cur["MultipleLines"]))

            st.markdown("##### 🔑 Google Gemini AI Strategy Configuration")
            gemini_api_key = st.text_input(
                "Enter Google Gemini API Key (Optional)",
                type="password",
                value=st.session_state.get("gemini_api_key", ""),
                placeholder="AIzaSy...",
                help="Securely enter your Google Gemini API key to activate Gemini 1.5 Flash personalized retention recommendations with custom discount offers."
            )
            st.session_state["gemini_api_key"] = gemini_api_key

            submit_predict = st.form_submit_button("🔮 Predict Churn & Reveal SHAP Drivers", type="primary", use_container_width=True)

    # ----------------------------------------------------
    # RIGHT: INSTINCTIVE INTELLIGENCE & SHAP CONSOLE
    # ----------------------------------------------------
    with viz_col:
        # Build payload
        payload = {
            "tenure": tenure,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges,
            "gender": gender,
            "SeniorCitizen": senior_citizen,
            "Partner": partner,
            "Dependents": dependents,
            "PhoneService": phone_service,
            "MultipleLines": multiple_lines,
            "InternetService": internet_service,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": cur.get("StreamingTV", "No"),
            "StreamingMovies": cur.get("StreamingMovies", "No"),
            "Contract": contract,
            "PaperlessBilling": paperless_billing,
            "PaymentMethod": payment_method,
            "gemini_api_key": st.session_state.get("gemini_api_key", "").strip()
        }
        st.session_state["form_data"] = payload

        # Always run or refresh prediction so user instantly sees results without searching
        res = None
        try:
            res = call_predict_churn_api(payload)
        except Exception as e:
            st.error(f"Inference gateway error: {e}")

        if res:
            prob = res["churn_probability"]
            tier = res["risk_tier"]
            pred = res["churn_prediction"]
            base_val = res.get("base_value", 0.0)
            contributions = res["feature_contributions"]
            retention_strategy = res.get("retention_strategy")

            # Gauge color dynamics with fresh vibrant energy
            if prob >= 0.65:
                accent_color = "#F43F5E"  # Electric Coral
                tier_badge = '<span class="badge-pill badge-danger">🚨 CRITICAL FLIGHT RISK</span>'
            elif prob >= 0.35:
                accent_color = "#F59E0B"  # Vibrant Sunfire Amber
                tier_badge = '<span class="badge-pill badge-warning">⚠️ MODERATE ATTENTION NEEDED</span>'
            else:
                accent_color = "#10B981"  # Spring Emerald Mint
                tier_badge = '<span class="badge-pill badge-success">🛡️ SECURE & LOYAL</span>'

            # Risk Score Speedometer / Gauge
            st.markdown(f"### 🎯 Predictive Churn Propensity: {tier_badge}", unsafe_allow_html=True)
            
            gauge_fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=prob * 100,
                number={"suffix": "%", "font": {"size": 48, "family": "Outfit", "color": "#F8FAFC"}},
                delta={"reference": 16.6, "increasing": {"color": "#F43F5E"}, "decreasing": {"color": "#10B981"}, "position": "top"},
                gauge={
                    "axis": {"range": [0, 100], "tickwidth": 1.5, "tickcolor": "#475569", "tickfont": {"size": 11, "color": "#94A3B8"}},
                    "bar": {"color": accent_color, "thickness": 0.35},
                    "bgcolor": "rgba(22, 30, 52, 0.5)",
                    "borderwidth": 0,
                    "steps": [
                        {"range": [0, 35], "color": "rgba(16, 185, 129, 0.18)"},
                        {"range": [35, 65], "color": "rgba(245, 158, 11, 0.18)"},
                        {"range": [65, 100], "color": "rgba(244, 63, 94, 0.22)"}
                    ],
                    "threshold": {
                        "line": {"color": "#FF2E63", "width": 4},
                        "thickness": 0.85,
                        "value": 50
                    }
                }
            ))
            gauge_fig.update_layout(
                height=265,
                margin=dict(l=25, r=25, t=25, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)"
            )
            st.plotly_chart(gauge_fig, use_container_width=True)

            # Gemini LLM Personalized Retention Strategy Display (Structured Enterprise SaaS Card)
            if retention_strategy:
                is_error = retention_strategy.startswith("⚠️")
                badge_class = "badge-danger" if is_error else "badge-info"
                badge_label = "API NOTICE" if is_error else "AI PRESCRIPTIVE PLAYBOOK"
                gemini_html = f"""
                <div class="gemini-card" style="background-color: #1E293B; border: 1px solid #334155; border-radius: 8px; padding: 1.4rem 1.5rem; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3), 0 2px 4px -1px rgba(0, 0, 0, 0.18); margin-top: 1.25rem; margin-bottom: 1.25rem;">
                    <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 0.85rem; border-bottom: 1px solid #334155; padding-bottom: 0.75rem; flex-wrap: wrap; gap: 8px;">
                        <div style="display: flex; align-items: center; gap: 10px;">
                            <span style="font-size: 1.35rem;">🤖</span>
                            <div>
                                <h4 style="margin: 0; color: #F8FAFC; font-family: 'Outfit', sans-serif; font-size: 1.12rem; font-weight: 700;">Executive Retention Playbook & Action Plan</h4>
                                <span style="font-size: 0.82rem; color: #94A3B8;">Powered by Google Gemini 1.5 Flash • Contextualized via customer SHAP log-odds</span>
                            </div>
                        </div>
                        <span class="badge-pill {badge_class}">{badge_label}</span>
                    </div>
                    <div style="color: #CBD5E1; font-size: 0.95rem; line-height: 1.65; white-space: pre-wrap; font-family: 'Plus Jakarta Sans', sans-serif;">
{retention_strategy}
                    </div>
                </div>
                """
                st.markdown(gemini_html, unsafe_allow_html=True)
            else:
                gemini_placeholder_html = """
                <div class="gemini-card" style="background-color: #1E293B; border: 1px solid #334155; border-radius: 8px; padding: 1.25rem 1.5rem; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3), 0 2px 4px -1px rgba(0, 0, 0, 0.18); margin-top: 1.25rem; margin-bottom: 1.25rem;">
                    <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 0.5rem;">
                        <span style="font-size: 1.3rem;">💡</span>
                        <h4 style="margin: 0; color: #F8FAFC; font-family: 'Outfit', sans-serif; font-size: 1.05rem; font-weight: 700;">Google Gemini Strategy Engine</h4>
                    </div>
                    <p style="color: #94A3B8; font-size: 0.9rem; line-height: 1.55; margin: 0;">
                        Enter your <b>Google Gemini API Key</b> in the left Customer Profile Designer panel to automatically generate an AI-tailored retention playbook with concrete operational intervention steps and targeted discount offers powered by <code>gemini-1.5-flash</code>.
                    </p>
                </div>
                """
                st.markdown(gemini_placeholder_html, unsafe_allow_html=True)

            # Prescriptive Actionable Playbook (Instinctive Guidance)
            top_risks = [c for c in contributions if c["contribution"] > 0][:3]
            top_protective = [c for c in contributions if c["contribution"] < 0][:3]

            if prob >= 0.50:
                rec_html = f"""
                <div class="rec-card-high">
                    <div style="font-weight: 800; color: #FDA4AF; font-size: 1.08rem; margin-bottom: 0.45rem; display: flex; align-items: center; gap: 8px;">
                        <span>⚡ Urgent Retention Intervention Recommended</span>
                    </div>
                    <div style="color: #E2E8F0; font-size: 0.92rem; line-height: 1.55;">
                        Primary flight catalyst is <b style="color: #FF80BF;">{top_risks[0]['feature'] if top_risks else 'high monthly charges'}</b> (+{top_risks[0]['contribution']:.2f} log-odds impact).<br>
                        <b>Prescriptive Playbook:</b> Deploy a targeted 15% rate lock incentive for switching to a 1-Year Agreement + bundle complimentary Tech Support.
                    </div>
                </div>
                """
            else:
                rec_html = f"""
                <div class="rec-card-low">
                    <div style="font-weight: 800; color: #A7F3D0; font-size: 1.08rem; margin-bottom: 0.45rem; display: flex; align-items: center; gap: 8px;">
                        <span>✅ Healthy Account Retention Dynamic</span>
                    </div>
                    <div style="color: #E2E8F0; font-size: 0.92rem; line-height: 1.55;">
                        Strongest loyalty driver is <b style="color: #34D399;">{top_protective[0]['feature'] if top_protective else 'contract lock-in'}</b> ({top_protective[0]['contribution']:.2f} log-odds impact).<br>
                        <b>Prescriptive Playbook:</b> Customer is primed for VIP loyalty cross-sell or premium speed tier upgrade.
                    </div>
                </div>
                """
            st.markdown(rec_html, unsafe_allow_html=True)

            # SHAP Waterfall Plot (Feature Contribution to Margin)
            raw_shap = res.get("raw_shap_values")
            feat_names = res.get("feature_names")
            if raw_shap and feat_names:
                st.markdown("<div style='height: 25px;'></div>", unsafe_allow_html=True)
                st.markdown("#### 🌊 SHAP Waterfall Plot (Feature Contribution to Margin)")
                st.caption("Visual decomposition of how individual customer attributes shift the model prediction away from the baseline E[f(x)] towards the churn prediction threshold.")

                try:
                    import shap
                    import matplotlib.pyplot as plt

                    explanation = shap.Explanation(
                        values=np.array(raw_shap),
                        base_values=float(base_val),
                        feature_names=feat_names
                    )

                    fig = plt.figure(figsize=(9, 5), facecolor='none')
                    shap.waterfall_plot(explanation, max_display=10, show=False)
                    fig = plt.gcf()
                    fig.patch.set_facecolor('none')
                    for ax in fig.axes:
                        ax.set_facecolor('none')
                        ax.tick_params(colors='#CBD5E1', labelsize=9)
                        ax.xaxis.label.set_color('#CBD5E1')
                        ax.yaxis.label.set_color('#CBD5E1')
                        for spine in ax.spines.values():
                            spine.set_color('#334155')
                        for t in ax.texts:
                            t.set_color('#CBD5E1')
                        for t in ax.get_xticklabels():
                            t.set_color('#CBD5E1')
                        for t in ax.get_yticklabels():
                            t.set_color('#CBD5E1')
                    plt.tight_layout()
                    st.pyplot(fig, clear_figure=True, transparent=True)
                    plt.close('all')
                except Exception as ex:
                    st.warning(f"Could not render SHAP waterfall plot: {str(ex)}")

            # Interactive SHAP TreeExplainer Contribution Chart
            st.markdown("<div style='height: 25px;'></div>", unsafe_allow_html=True)
            st.markdown("#### 🔬 Explainable SHAP Attribution (Why did the AI score this?)")
            st.caption("🔴 Red bars push towards Churn | 🟢 Green bars push towards Loyalty Retention")

            top_contributions = contributions[:12]
            df_shap = pd.DataFrame(top_contributions).iloc[::-1]

            bar_colors = ["#F43F5E" if v > 0 else "#10B981" for v in df_shap["contribution"]]

            shap_fig = go.Figure(go.Bar(
                y=df_shap["feature"],
                x=df_shap["contribution"],
                orientation="h",
                marker=dict(color=bar_colors, line=dict(color="rgba(255,255,255,0.15)", width=1)),
                text=[f"{v:+.2f}" for v in df_shap["contribution"]],
                textposition="outside",
                textfont=dict(color="#F8FAFC", size=11, family="Outfit"),
                hovertemplate="<b>%{y}</b><br>SHAP Log-Odds Impact: %{x:+.4f}<extra></extra>"
            ))
            shap_fig.update_layout(
                height=370,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(10, 15, 28, 0.6)",
                font=dict(color="#94A3B8"),
                xaxis=dict(
                    title="SHAP Value (Impact on Model Margin)",
                    gridcolor="rgba(255,255,255,0.07)",
                    zeroline=True,
                    zerolinecolor="#38BDF8",
                    zerolinewidth=2
                ),
                yaxis=dict(gridcolor="rgba(0,0,0,0)"),
                margin=dict(l=140, r=40, t=15, b=40)
            )
            st.plotly_chart(shap_fig, use_container_width=True)

    # ----------------------------------------------------
    # MACRO SURVIVAL CURVES (SEAMLESSLY INTEGRATED ON PAGE 1)
    # ----------------------------------------------------
    st.markdown("---")
    st.markdown("### 📈 Macro Cohort Survival Dynamics (Kaplan-Meier)")
    st.caption("Empirical survival probability over time: S(t) = P(Tenure > t), incorporating right-censored accounts across 100,000 subscriber lifelines.")

    strat_col1, strat_col2 = st.columns([2, 1])
    with strat_col1:
        view_mode = st.radio(
            "Stratify Survival Function By:",
            ["Overall Cohort", "Contract Type (Month-to-month vs 1-Year vs 2-Year)", "Internet Service Tier"],
            horizontal=True
        )

    fig_surv = go.Figure()

    if view_mode == "Overall Cohort":
        ci_lower = np.array(km_data["overall"]["ci_lower"])
        ci_upper = np.array(km_data["overall"]["ci_upper"])
        fig_surv.add_trace(go.Scatter(
            x=list(timeline) + list(timeline[::-1]),
            y=list(ci_upper) + list(ci_lower[::-1]),
            fill='toself',
            fillcolor='rgba(0, 242, 254, 0.18)',
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip",
            name="95% Confidence Interval"
        ))
        fig_surv.add_trace(go.Scatter(
            x=timeline,
            y=survival_prob,
            mode="lines",
            name="Overall S(t)",
            line=dict(color="#00F2FE", width=4),
            hovertemplate="<b>Tenure:</b> %{x} months<br><b>Survival Probability:</b> %{y:.2%}<extra></extra>"
        ))
        fig_surv.add_hline(y=0.5, line_dash="dash", line_color="#F43F5E", annotation_text="Median Survival (S=0.50)", annotation_position="bottom right")

    elif view_mode == "Contract Type (Month-to-month vs 1-Year vs 2-Year)":
        colors = {"Month-to-month": "#F43F5E", "One year": "#F59E0B", "Two year": "#10B981"}
        for c_type, data in km_data.get("by_contract", {}).items():
            fig_surv.add_trace(go.Scatter(
                x=data["timeline"],
                y=data["survival_prob"],
                mode="lines",
                name=f"{c_type} Contract",
                line=dict(color=colors.get(c_type, "#6B7280"), width=3.8),
                hovertemplate=f"<b>{c_type}</b><br>Tenure: %{{x}} mo<br>Survival: %{{y:.2%}}<extra></extra>"
            ))
        fig_surv.add_hline(y=0.5, line_dash="dash", line_color="#64748B")

    elif view_mode == "Internet Service Tier":
        colors = {"Fiber optic": "#EC4899", "DSL": "#00F2FE", "No": "#10B981"}
        for inet, data in km_data.get("by_internet", {}).items():
            fig_surv.add_trace(go.Scatter(
                x=data["timeline"],
                y=data["survival_prob"],
                mode="lines",
                name=f"Internet: {inet}",
                line=dict(color=colors.get(inet, "#6B7280"), width=3.8),
                hovertemplate=f"<b>Internet: {inet}</b><br>Tenure: %{{x}} mo<br>Survival: %{{y:.2%}}<extra></extra>"
            ))
        fig_surv.add_hline(y=0.5, line_dash="dash", line_color="#64748B")

    fig_surv.update_layout(
        height=450,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(10, 15, 28, 0.6)",
        font=dict(color="#94A3B8"),
        xaxis=dict(title="Tenure Elapsed (Months)", gridcolor="rgba(255,255,255,0.07)", range=[0, 74], dtick=6),
        yaxis=dict(title="Survival Probability S(t)", gridcolor="rgba(255,255,255,0.07)", tickformat=".0%", range=[0, 1.02]),
        hovermode="x unified",
        legend=dict(bgcolor="rgba(15, 23, 42, 0.8)", bordercolor="rgba(255,255,255,0.12)", borderwidth=1),
        margin=dict(l=60, r=30, t=30, b=50)
    )
    st.plotly_chart(fig_surv, use_container_width=True)

    with st.expander("📋 View Complete Feature Attribution Table for Current Profile"):
        if res:
            st.dataframe(pd.DataFrame(res["feature_contributions"]), use_container_width=True)

# ========================================================
# TAB 2: ENTERPRISE BATCH PREDICTION
# ========================================================
with tab2:
    st.subheader("📁 High-Throughput Streaming Batch Engine")
    st.caption("Engineered for datasets up to **1,000,000+ rows and 1,000+ columns** in CSV, Parquet, JSON, JSONL, and Feather.")

    top_col1, top_col2, top_col3 = st.columns([2, 1.5, 1.5])
    
    with top_col1:
        fmt_choice = st.selectbox(
            "Desired Export Format:",
            ["Match Input Format", "Parquet (.parquet - Recommended for >100k rows)", "CSV (.csv)", "JSON Lines (.jsonl)"],
            index=0
        )
        format_param_map = {
            "Match Input Format": "auto",
            "Parquet (.parquet - Recommended for >100k rows)": "parquet",
            "CSV (.csv)": "csv",
            "JSON Lines (.jsonl)": "jsonl"
        }
        selected_fmt = format_param_map[fmt_choice]

    with top_col2:
        if os.path.exists("telco_data.csv"):
            sample_df = pd.read_csv("telco_data.csv").head(100).drop(columns=["Churn"], errors="ignore")
            st.download_button(
                label="📥 Download Sample CSV",
                data=sample_df.to_csv(index=False).encode("utf-8"),
                file_name="sample_customer_batch.csv",
                mime="text/csv",
                use_container_width=True
            )

    with top_col3:
        test_synthetic = st.button("⚡ Test 25k Rows × 100 Columns", use_container_width=True, help="Generates an in-memory 25,000 rows × 100 columns Parquet dataset and runs it through the batch engine.")

    uploaded_file = st.file_uploader(
        "Upload Customer Dataset",
        type=["csv", "parquet", "pq", "json", "jsonl", "ndjson", "feather", "arrow"],
        help="Upload files up to 1,000,000+ rows and 1,000+ columns. Supported: CSV, Parquet, JSON, JSONL, Feather."
    )

    if test_synthetic:
        with st.spinner("Generating 25,000 rows × 100 columns synthetic dataset in Parquet format..."):
            import pyarrow as pa
            import pyarrow.parquet as pq
            from io import BytesIO
            
            n_synth = 25_000
            synth_dict = {f"behavioral_feature_{i}": np.random.randn(n_synth).astype(np.float32) for i in range(85)}
            synth_dict["customerID"] = [f"SYNTH-{i:07d}" for i in range(n_synth)]
            synth_dict["tenure"] = np.random.randint(1, 72, size=n_synth)
            synth_dict["MonthlyCharges"] = np.random.uniform(20.0, 115.0, size=n_synth).round(2)
            synth_dict["TotalCharges"] = (synth_dict["tenure"] * synth_dict["MonthlyCharges"]).round(2)
            synth_dict["Contract"] = np.random.choice(["Month-to-month", "One year", "Two year"], size=n_synth)
            synth_dict["InternetService"] = np.random.choice(["Fiber optic", "DSL", "No"], size=n_synth)
            synth_dict["PaymentMethod"] = np.random.choice(["Electronic check", "Mailed check", "Bank transfer (automatic)"], size=n_synth)
            synth_dict["TechSupport"] = np.random.choice(["Yes", "No"], size=n_synth)
            synth_dict["OnlineSecurity"] = np.random.choice(["Yes", "No"], size=n_synth)
            synth_df = pd.DataFrame(synth_dict)
            
            pq_buf = BytesIO()
            synth_df.to_parquet(pq_buf, index=False)
            synthetic_bytes = pq_buf.getvalue()
            
            class MockUploadedFile:
                def __init__(self, name: str, data: bytes):
                    self.name = name
                    self.data = data
                def getvalue(self) -> bytes:
                    return self.data
            
            uploaded_file = MockUploadedFile("synthetic_25k_100cols.parquet", synthetic_bytes)

    if uploaded_file is not None:
        file_bytes = uploaded_file.getvalue()
        filename = uploaded_file.name
        file_size_mb = len(file_bytes) / (1024 * 1024)

        st.info(f"📄 **Target File:** `{filename}` | **Size:** `{file_size_mb:.2f} MB`")

        with st.spinner("Streaming data chunks through XGBoost inference engine..."):
            start_t = time.time()
            try:
                scored_bytes, preview_df, stats, effective_fmt = call_predict_batch_api(file_bytes, filename, selected_fmt)
                elapsed_s = max(time.time() - start_t, 0.001)
            except Exception as e:
                st.error(f"Batch processing error: {str(e)}")
                scored_bytes, preview_df, stats = None, None, None

        if stats is not None and preview_df is not None:
            total_rows = stats.get("total_rows", 0)
            total_cols = stats.get("total_columns", 0)
            throughput = int(total_rows / elapsed_s)

            st.success(f"⚡ Successfully scored **{total_rows:,}** rows across **{total_cols:,}** columns in **{elapsed_s:.2f}s** ({throughput:,} rows/sec)!")

            b_c1, b_c2, b_c3, b_c4 = st.columns(4)
            avg_risk = stats.get("mean_risk_score", 0.0) * 100
            high_count = stats.get("high_risk_count", 0)
            low_count = stats.get("low_risk_count", max(0, total_rows - high_count))

            with b_c1:
                st.metric("Total Records Scored", f"{total_rows:,}", f"{total_cols} cols")
            with b_c2:
                st.metric("Mean Churn Risk", f"{avg_risk:.1f}%")
            with b_c3:
                st.metric("🚨 High Risk Cohort", f"{high_count:,}", f"{(high_count/max(total_rows,1)*100):.1f}%")
            with b_c4:
                st.metric("🛡️ Low Risk Cohort", f"{low_count:,}", f"{(low_count/max(total_rows,1)*100):.1f}%")

            if "Churn_Risk_Score" in preview_df.columns:
                fig_hist = px.histogram(
                    preview_df,
                    x="Churn_Risk_Score",
                    nbins=30,
                    color_discrete_sequence=["#06B6D4"],
                    title="<b>Risk Score Distribution (Sampled Preview)</b>",
                    labels={"Churn_Risk_Score": "Churn Risk Probability"}
                )
                fig_hist.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(10, 15, 28, 0.6)",
                    height=280,
                    margin=dict(l=40, r=40, t=50, b=40),
                    xaxis=dict(range=[0, 1], gridcolor="rgba(255,255,255,0.07)"),
                    yaxis=dict(gridcolor="rgba(255,255,255,0.07)")
                )
                st.plotly_chart(fig_hist, use_container_width=True)

            st.write(f"#### 📋 Scored Records Preview ({min(len(preview_df), 500)} rows shown)")
            st.caption(f"Preview across all {total_cols} columns with appended `Churn_Risk_Score`.")
            st.dataframe(preview_df, use_container_width=True, height=420)

            mime_map = {
                "csv": "text/csv",
                "parquet": "application/vnd.apache.parquet",
                "json": "application/json",
                "jsonl": "application/x-ndjson",
                "feather": "application/octet-stream"
            }
            base_name = os.path.splitext(filename)[0]
            download_filename = f"scored_{base_name}.{effective_fmt}"

            st.download_button(
                label=f"📥 Download Full Scored Dataset ({effective_fmt.upper()})",
                data=scored_bytes,
                file_name=download_filename,
                mime=mime_map.get(effective_fmt, "application/octet-stream"),
                type="primary",
                use_container_width=True
            )

# ========================================================
# TAB 3: ANALYTICAL DATABASE & SQL WORKBENCH (DUCKDB)
# ========================================================
with tab3:
    st.subheader("🗄️ DuckDB Vectorized Analytical Database & Cohort Intelligence")
    st.caption("Embedded columnar OLAP database handling out-of-core persistence, sub-millisecond aggregations on 100k+ customer lifelines, and real-time inference auditing.")

    db_overview = None
    try:
        import requests
        resp = requests.get(f"{BACKEND_URL}/db/overview", timeout=5)
        if resp.status_code == 200:
            db_overview = resp.json()
    except Exception:
        try:
            from db import get_database_overview
            db_overview = get_database_overview()
        except Exception:
            pass

    if db_overview:
        dcol1, dcol2, dcol3, dcol4 = st.columns(4)
        with dcol1:
            st.markdown(f"""
            <div class="kpi-card kpi-card-cyan">
                <div class="kpi-label">Database Engine</div>
                <div class="kpi-value" style="font-size: 1.5rem; color: #00F2FE;">DuckDB OLAP</div>
                <div class="kpi-sub">Vectorized C++ Engine</div>
            </div>
            """, unsafe_allow_html=True)
        with dcol2:
            st.markdown(f"""
            <div class="kpi-card kpi-card-mint">
                <div class="kpi-label">Master Records</div>
                <div class="kpi-value" style="color: #10B981;">{db_overview['total_master_customers']:,}</div>
                <div class="kpi-sub">table: customers</div>
            </div>
            """, unsafe_allow_html=True)
        with dcol3:
            st.markdown(f"""
            <div class="kpi-card kpi-card-indigo">
                <div class="kpi-label">Storage Footprint</div>
                <div class="kpi-value" style="color: #818CF8;">{db_overview['db_size_mb']} MB</div>
                <div class="kpi-sub" style="color: #A5B4FC;">Columnar Compressed</div>
            </div>
            """, unsafe_allow_html=True)
        with dcol4:
            st.markdown(f"""
            <div class="kpi-card kpi-card-coral">
                <div class="kpi-label">Audit Logged</div>
                <div class="kpi-value" style="color: #F43F5E;">{db_overview['total_predictions_persisted']:,}</div>
                <div class="kpi-sub" style="color: #FDA4AF;">table: predictions_audit</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        st.write("### 📊 Macro Survival Cohorts Matrix (DuckDB View)")
        st.caption("Vectorized aggregations across contract durations and connectivity tiers (`v_survival_cohorts`).")

        cohort_data = db_overview.get("survival_cohorts", [])
        if cohort_data:
            df_cohorts = pd.DataFrame(cohort_data)
            
            fig_cohort = px.bar(
                df_cohorts,
                x="Contract",
                y="churn_rate_pct",
                color="InternetService",
                barmode="group",
                text="churn_rate_pct",
                title="<b>Empirical Churn Rate (%) by Contract Type & Internet Service Tier</b>",
                labels={"churn_rate_pct": "Churn Rate (%)", "Contract": "Contract Type"},
                color_discrete_map={"Fiber optic": "#EC4899", "DSL": "#00F2FE", "No": "#10B981"}
            )
            fig_cohort.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
            fig_cohort.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(10, 15, 28, 0.6)",
                height=380,
                yaxis=dict(ticksuffix="%", gridcolor="rgba(255,255,255,0.07)"),
                xaxis=dict(gridcolor="rgba(255,255,255,0.07)")
            )
            st.plotly_chart(fig_cohort, use_container_width=True)

            with st.expander("📋 View Complete Cohort Aggregate Life Table"):
                st.dataframe(df_cohorts, use_container_width=True)

        st.markdown("---")

        st.write("### 🚨 Persistent High-Risk Customer Watchlist")
        st.caption("Live query from `predictions_audit` table tracking all past scored accounts.")

        top_risks = []
        try:
            r_resp = requests.get(f"{BACKEND_URL}/db/top-risks?limit=50", timeout=5)
            if r_resp.status_code == 200:
                top_risks = r_resp.json()
        except Exception:
            try:
                from db import query_top_risk_accounts
                top_risks = query_top_risk_accounts(limit=50)
            except Exception:
                pass

        if top_risks:
            df_top = pd.DataFrame(top_risks)
            st.dataframe(df_top, use_container_width=True, height=280)
        else:
            st.info("ℹ️ No historical inference records logged yet. Predictions will automatically audit here as you run single and batch inference!")

        st.markdown("---")

        st.write("### ⚡ DuckDB SQL Analytics Workbench")
        st.caption("Run blazing-fast analytical SQL directly on the embedded database tables: `customers`, `predictions_audit`, `v_survival_cohorts`.")

        pcol1, pcol2, pcol3, pcol4 = st.columns(4)
        selected_sql = "SELECT Contract, COUNT(*) as subscribers, ROUND(AVG(tenure), 1) as avg_tenure, ROUND(AVG(MonthlyCharges), 2) as avg_charges, ROUND(AVG(Churn)*100, 2) as churn_rate_pct FROM customers GROUP BY Contract ORDER BY churn_rate_pct DESC;"

        with pcol1:
            if st.button("💳 Payment Method Churn"):
                st.session_state["active_sql"] = "SELECT PaymentMethod, COUNT(*) as count, ROUND(AVG(tenure), 1) as avg_tenure, ROUND(AVG(Churn)*100, 2) as churn_pct FROM customers GROUP BY PaymentMethod ORDER BY churn_pct DESC;"
        with pcol2:
            if st.button("👴 Senior Citizen Profile"):
                st.session_state["active_sql"] = "SELECT SeniorCitizen, COUNT(*) as count, ROUND(AVG(MonthlyCharges), 2) as avg_monthly, ROUND(AVG(Churn)*100, 2) as churn_pct FROM customers GROUP BY SeniorCitizen;"
        with pcol3:
            if st.button("🌐 Tech Support & Churn"):
                st.session_state["active_sql"] = "SELECT TechSupport, InternetService, COUNT(*) as subscribers, ROUND(AVG(Churn)*100, 2) as churn_pct FROM customers WHERE InternetService != 'No' GROUP BY TechSupport, InternetService ORDER BY churn_pct DESC;"
        with pcol4:
            if st.button("📈 High-Spend Cohorts (> $85)"):
                st.session_state["active_sql"] = "SELECT Contract, COUNT(*) as high_spenders, ROUND(AVG(Churn)*100, 2) as churn_pct FROM customers WHERE MonthlyCharges > 85 GROUP BY Contract ORDER BY churn_pct DESC;"

        current_query = st.session_state.get("active_sql", selected_sql)
        sql_input = st.text_area("SQL Query Input (DuckDB Vectorized SQL):", value=current_query, height=100)

        if st.button("🚀 Execute Analytical SQL Query", type="primary"):
            with st.spinner("Executing vectorized query via DuckDB engine..."):
                query_res = None
                try:
                    q_resp = requests.post(f"{BACKEND_URL}/db/query", json={"query": sql_input}, timeout=10)
                    if q_resp.status_code == 200:
                        query_res = q_resp.json()
                    else:
                        st.error(f"Query Error: {q_resp.text}")
                except Exception as e:
                    try:
                        from db import run_custom_sql
                        res_df = run_custom_sql(sql_input)
                        query_res = {"columns": list(res_df.columns), "rows_count": len(res_df), "records": res_df.to_dict(orient="records")}
                    except Exception as ex:
                        st.error(f"Execution Error: {str(ex)}")

                if query_res and query_res.get("records"):
                    res_df = pd.DataFrame(query_res["records"])
                    st.success(f"Query returned **{query_res['rows_count']:,}** rows.")
                    st.dataframe(res_df, use_container_width=True)
                    
                    st.download_button(
                        label="📥 Download Query Results as CSV",
                        data=res_df.to_csv(index=False).encode("utf-8"),
                        file_name="duckdb_query_result.csv",
                        mime="text/csv"
                    )
    else:
        st.warning("⚠️ Could not connect to DuckDB analytical database. Please ensure `backend.py` is running.")
