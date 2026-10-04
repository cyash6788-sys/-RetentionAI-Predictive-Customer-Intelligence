# ⚡ RetentionAI: Predictive Customer Intelligence Platform

An enterprise-grade customer retention and churn intelligence platform powered by **XGBoost**, **Tree SHAP**, **DuckDB columnar OLAP**, **Kaplan-Meier survival analysis**, and **Google Gemini 1.5 Flash**.

---

## 🌟 Key Features

1. **🎯 Micro Customer Risk Profiling & Interactive Designer**:
   - Real-time customer churn probability scoring (< 8ms latency).
   - One-click archetype presets (High Flight Risk, Low Risk Loyal, Mid-Tenure At-Risk).
   - Dynamic Propensity Speedometer gauge with segmented hazard tiers.

2. **🤖 Google Gemini 1.5 Flash Retention Playbook**:
   - Hyper-personalized AI retention strategies generated on-the-fly.
   - Dynamic prompting tailored directly from the customer's top 3 SHAP log-odds drivers.
   - Prescriptive operational steps and targeted rate-lock / discount offers.

3. **🌊 Explainable AI (XAI) Attribution & Waterfall Plots**:
   - Transparent Matplotlib waterfall plots decomposing model margins from baseline $E[f(x)]$ to individual prediction thresholds.
   - Interactive horizontal bar charts attributing positive (flight catalyst) vs. negative (protective loyalty) feature contributions.

4. **📈 Macro Cohort Survival Dynamics (Kaplan-Meier)**:
   - Empirical survival probability curves ($S(t) = P(\text{Tenure} > t)$) accounting for right-censored lifelines across 100,000 subscribers.
   - Stratified breakdowns by Contract Duration (Month-to-month vs. 1-Year vs. 2-Year) and Internet Service Tiers (Fiber optic vs. DSL).

5. **📁 High-Throughput Streaming Batch Engine**:
   - Streaming chunk-based inference capable of handling datasets with **1,000,000+ rows and 1,000+ columns**.
   - Supports **CSV, Parquet, JSON, JSONL, and Feather** formats with automated output type conversion and summary statistics.

6. **🗄️ DuckDB Vectorized Analytical Database & Audit Log**:
   - Embedded columnar OLAP database handling sub-millisecond cohort aggregations.
   - Full inference audit logging with table views for high-risk accounts and retention cohort matrices.

---

## 🛠️ Architecture & Tech Stack

- **ML Inference**: Scikit-Learn, XGBoost (`XGBClassifier`), SHAP (`TreeExplainer`), Lifelines (`KaplanMeierFitter`)
- **Backend API**: FastAPI, Uvicorn, Pydantic, PyArrow
- **Analytical Storage**: DuckDB Vectorized Columnar Database
- **Frontend Dashboard**: Streamlit, Plotly Express/Graph Objects, Matplotlib
- **Generative AI**: Google Gemini API (`gemini-1.5-flash`) via `google-generativeai`

---

## 🚀 Quick Start Guide

### 1. Prerequisites & Environment Setup
```bash
# Clone the repository
git clone https://github.com/cyash6788-sys/-RetentionAI-Predictive-Customer-Intelligence.git
cd -RetentionAI-Predictive-Customer-Intelligence

# Create and activate virtual environment
python -m venv .venv
# Windows (PowerShell):
.\.venv\Scripts\Activate.ps1
# Linux/macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Generate Synthetic Dataset & Train Models (Optional)
Pre-trained model artifacts are included in the `artifacts/` folder. To regenerate fresh data and retrain:
```bash
python generate_telco_data.py
python model_engine.py
python db.py
```

### 3. Launch the FastAPI Backend
```bash
python -m uvicorn backend:app --host 127.0.0.1 --port 8000
```
Backend API docs available at `http://127.0.0.1:8000/docs`.

### 4. Launch the Streamlit Frontend
```bash
python -m streamlit run frontend.py --server.port 8501
```
Open your browser at `http://localhost:8501`.

---

## 🔑 Gemini AI Strategy Configuration
To enable automated Gemini AI retention recommendations, enter your Google Gemini API Key directly in the password field in the Customer Profile Designer panel on Tab 1.
