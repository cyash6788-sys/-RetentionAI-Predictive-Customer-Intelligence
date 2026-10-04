"""
FastAPI Backend for Customer Churn Prediction & Survival Insights
- Endpoint /predict-churn: Preprocesses customer data, predicts churn probability with XGBoost,
  and uses shap.TreeExplainer to compute exact feature contributions.
- Endpoint /survival-macro: Returns Kaplan-Meier macro survival distributions.
- Endpoint /health: Health check and model metadata.
"""

import os
from contextlib import asynccontextmanager
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import joblib
import tempfile
from io import BytesIO
from fastapi import FastAPI, HTTPException, UploadFile, File, Query, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import shap
from batch_processor import process_large_dataset, detect_file_format
from db import init_db, get_database_overview, query_top_risk_accounts, run_custom_sql, save_predictions_audit

# Paths to saved artifacts
ARTIFACTS_DIR = "artifacts"
PREPROCESSOR_PATH = os.path.join(ARTIFACTS_DIR, "preprocessor.joblib")
XGB_MODEL_PATH = os.path.join(ARTIFACTS_DIR, "xgb_model.joblib")
METADATA_PATH = os.path.join(ARTIFACTS_DIR, "metadata.joblib")
KM_DATA_PATH = os.path.join(ARTIFACTS_DIR, "km_data.joblib")

# Global model state
models: Dict[str, Any] = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize DuckDB analytical database
    try:
        init_db()
        print("DuckDB analytical database connected and initialized.")
    except Exception as e:
        print(f"DuckDB initialization warning: {e}")

    # Load models and explainer on startup
    if not os.path.exists(XGB_MODEL_PATH) or not os.path.exists(PREPROCESSOR_PATH):
        print("Warning: Model artifacts not found yet. Run model_engine.py first.")
    else:
        print("Loading model artifacts into memory...")
        models["preprocessor"] = joblib.load(PREPROCESSOR_PATH)
        models["xgb_model"] = joblib.load(XGB_MODEL_PATH)
        models["metadata"] = joblib.load(METADATA_PATH)
        if os.path.exists(KM_DATA_PATH):
            models["km_data"] = joblib.load(KM_DATA_PATH)
        
        # Initialize SHAP TreeExplainer
        print("Initializing SHAP TreeExplainer...")
        models["explainer"] = shap.TreeExplainer(models["xgb_model"])
        print("Backend ready to serve predictions.")
    yield
    models.clear()

app = FastAPI(
    title="RetentionAI: Predictive Customer Intelligence API",
    description="Real-time churn risk scoring, explainable SHAP attribution, Kaplan-Meier survival dynamics, and DuckDB OLAP engine powered by XGBoost",
    version="2.5.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class CustomerProfile(BaseModel):
    tenure: int = Field(default=12, ge=1, le=100, description="Tenure in months")
    MonthlyCharges: float = Field(default=70.0, ge=10.0, le=200.0, description="Monthly charges in USD")
    TotalCharges: float = Field(default=840.0, ge=0.0, description="Total cumulative charges in USD")
    gender: str = Field(default="Female", description="Gender (Male / Female)")
    SeniorCitizen: int = Field(default=0, ge=0, le=1, description="Senior Citizen flag (0 or 1)")
    Partner: str = Field(default="No", description="Has partner (Yes / No)")
    Dependents: str = Field(default="No", description="Has dependents (Yes / No)")
    PhoneService: str = Field(default="Yes", description="Has phone service (Yes / No)")
    MultipleLines: str = Field(default="No", description="Multiple lines (Yes / No / No phone service)")
    InternetService: str = Field(default="Fiber optic", description="Internet service type (DSL / Fiber optic / No)")
    OnlineSecurity: str = Field(default="No", description="Online security (Yes / No / No internet service)")
    OnlineBackup: str = Field(default="No", description="Online backup (Yes / No / No internet service)")
    DeviceProtection: str = Field(default="No", description="Device protection (Yes / No / No internet service)")
    TechSupport: str = Field(default="No", description="Tech support (Yes / No / No internet service)")
    StreamingTV: str = Field(default="Yes", description="Streaming TV (Yes / No / No internet service)")
    StreamingMovies: str = Field(default="Yes", description="Streaming Movies (Yes / No / No internet service)")
    Contract: str = Field(default="Month-to-month", description="Contract duration (Month-to-month / One year / Two year)")
    PaperlessBilling: str = Field(default="Yes", description="Paperless billing (Yes / No)")
    PaymentMethod: str = Field(default="Electronic check", description="Payment method")
    gemini_api_key: Optional[str] = Field(default=None, description="Google Gemini API Key for personalized retention strategy generation")

class FeatureContribution(BaseModel):
    feature: str
    contribution: float
    abs_contribution: float
    description: str

class ChurnPredictionResponse(BaseModel):
    churn_probability: float
    churn_prediction: int
    risk_tier: str
    base_value: float
    total_shap_margin: float
    feature_contributions: List[FeatureContribution]
    raw_shap_values: Optional[List[float]] = None
    feature_names: Optional[List[str]] = None
    retention_strategy: Optional[str] = None
    input_summary: Dict[str, Any]

@app.get("/health")
def health():
    return {
        "status": "healthy" if "xgb_model" in models else "degraded",
        "models_loaded": list(models.keys()),
        "roc_auc": models.get("metadata", {}).get("roc_auc")
    }

@app.get("/survival-macro")
def get_survival_macro():
    if "km_data" not in models and os.path.exists(KM_DATA_PATH):
        models["km_data"] = joblib.load(KM_DATA_PATH)
    if "km_data" not in models:
        raise HTTPException(status_code=404, detail="Kaplan-Meier survival data not found.")
    return models["km_data"]

class SQLQueryRequest(BaseModel):
    query: str

@app.get("/db/overview")
def get_db_overview_endpoint():
    try:
        return get_database_overview()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

@app.get("/db/top-risks")
def get_db_top_risks(limit: int = 50):
    try:
        return query_top_risk_accounts(limit=limit)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database query error: {str(e)}")

@app.post("/db/query")
def run_sql_query(req: SQLQueryRequest):
    try:
        clean_q = req.query.strip()
        if not clean_q.upper().startswith(("SELECT", "WITH", "DESCRIBE", "EXPLAIN", "SHOW", "PRAGMA")):
            raise HTTPException(status_code=400, detail="Only analytical read-only SQL queries (SELECT, WITH, DESCRIBE, SHOW) are permitted.")
        df = run_custom_sql(clean_q)
        return {
            "columns": list(df.columns),
            "rows_count": len(df),
            "records": df.head(1000).to_dict(orient="records")
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"DuckDB SQL Execution Error: {str(e)}")

@app.post("/predict-churn", response_model=ChurnPredictionResponse)
def predict_churn(customer: CustomerProfile):
    # Ensure models are loaded
    if "xgb_model" not in models:
        if os.path.exists(XGB_MODEL_PATH) and os.path.exists(PREPROCESSOR_PATH):
            models["preprocessor"] = joblib.load(PREPROCESSOR_PATH)
            models["xgb_model"] = joblib.load(XGB_MODEL_PATH)
            models["metadata"] = joblib.load(METADATA_PATH)
            models["explainer"] = shap.TreeExplainer(models["xgb_model"])
        else:
            raise HTTPException(status_code=503, detail="Models are not trained yet. Run model_engine.py first.")

    preprocessor = models["preprocessor"]
    xgb_model = models["xgb_model"]
    explainer = models["explainer"]
    metadata = models["metadata"]
    feature_names = metadata["transformed_feature_names"]

    # Convert customer to DataFrame
    input_dict = customer.model_dump()
    input_df = pd.DataFrame([input_dict])

    # Transform features
    try:
        X_trans = preprocessor.transform(input_df)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Preprocessing error: {str(e)}")

    # Predict probability
    prob_array = xgb_model.predict_proba(X_trans)[0]
    churn_prob = float(prob_array[1])
    churn_pred = int(churn_prob >= 0.5)

    # Risk tier classification
    if churn_prob >= 0.65:
        risk_tier = "High Risk"
    elif churn_prob >= 0.35:
        risk_tier = "Moderate Risk"
    else:
        risk_tier = "Low Risk"

    # SHAP tree explanation
    shap_vals = explainer.shap_values(X_trans)
    
    # Handle SHAP output shapes
    if isinstance(shap_vals, list):
        # Multiclass or binary with 2 outputs
        sample_shap = np.array(shap_vals[1][0], dtype=float)
    else:
        # Binary margin output
        sample_shap = np.array(shap_vals[0], dtype=float)

    # Base value
    base_val = explainer.expected_value
    if isinstance(base_val, (list, np.ndarray)):
        base_val = float(base_val[-1])
    else:
        base_val = float(base_val)

    # Compile feature contributions
    contributions = []
    for feat_name, shap_val in zip(feature_names, sample_shap):
        impact_dir = "Increases Churn Risk" if shap_val > 0 else "Decreases Churn Risk (Protective)"
        contributions.append({
            "feature": feat_name,
            "contribution": round(float(shap_val), 4),
            "abs_contribution": round(float(abs(shap_val)), 4),
            "description": f"{impact_dir} by {abs(shap_val):.3f}"
        })

    # Sort by absolute impact
    contributions.sort(key=lambda x: x["abs_contribution"], reverse=True)

    total_margin = float(np.sum(sample_shap) + base_val)

    # ----------------------------------------------------
    # GEMINI AI RETENTION STRATEGY (gemini-1.5-flash)
    # ----------------------------------------------------
    retention_strategy = None
    gemini_key = (customer.gemini_api_key or os.environ.get("GEMINI_API_KEY", "")).strip()

    if gemini_key:
        try:
            import google.generativeai as genai
            genai.configure(api_key=gemini_key)
            gemini_model = genai.GenerativeModel("gemini-1.5-flash")

            top_3_drivers = contributions[:3]
            top_3_text = "\n".join([
                f"- **{driver['feature']}**: {driver['description']} (SHAP value: {driver['contribution']:+.3f})"
                for driver in top_3_drivers
            ])

            prompt = f"""You are RetentionAI, an expert Chief Customer Officer and AI Retention Specialist.
A customer has been analyzed using an calibrated XGBoost model with Explainable SHAP TreeExplainer:

**Customer Risk Profile:**
- **Churn Risk Score (Probability):** {churn_prob:.1%}
- **Risk Tier:** {risk_tier}
- **Tenure:** {customer.tenure} months
- **Monthly Charges:** ${customer.MonthlyCharges:.2f}
- **Contract Type:** {customer.Contract}
- **Internet Service:** {customer.InternetService}
- **Payment Method:** {customer.PaymentMethod}

**Top 3 Highest-Contributing SHAP Risk Drivers:**
{top_3_text}

**Your Task:**
Provide a structured, personalized customer retention strategy specifically tailored to counteract the top SHAP drivers above. Format your response cleanly using markdown headings and bullet points:

### 1. 🔍 Root-Cause Diagnostic
Summarize in 1-2 concise sentences why this customer is at flight risk based on their top 3 SHAP features.

### 2. ⚡ Recommended Retention Action
Detail concrete, immediate operational steps for customer success or support representatives to execute.

### 3. 🎁 Tailored Financial Incentive & Discount Offer
Propose a specific, commercially viable discount offer or fee adjustment (e.g., specific percentage discount for X months, bill credit, waiver) designed to neutralize their churn propensity.

### 4. 🚀 Long-Term Value & Stickiness Plan
Recommend relevant service bundles or engagement milestones to drive retention beyond the incentive period.
"""

            response = gemini_model.generate_content(prompt)
            if response and response.text:
                retention_strategy = response.text.strip()
        except Exception as e:
            retention_strategy = f"⚠️ Gemini API Notice: Unable to generate strategy with provided key ({str(e)}). Please verify your Google Gemini API Key."

    # Persist prediction to DuckDB audit log (strip API key for privacy/security)
    clean_audit_dict = {k: v for k, v in input_dict.items() if k != "gemini_api_key"}
    try:
        audit_record = pd.DataFrame([{**clean_audit_dict, "Churn_Risk_Score": churn_prob}])
        save_predictions_audit(audit_record, batch_id="single-profile")
    except Exception as e:
        print(f"Audit log warning: {e}")

    return ChurnPredictionResponse(
        churn_probability=round(churn_prob, 4),
        churn_prediction=churn_pred,
        risk_tier=risk_tier,
        base_value=round(base_val, 4),
        total_shap_margin=round(total_margin, 4),
        feature_contributions=contributions,
        raw_shap_values=[round(float(v), 5) for v in sample_shap],
        feature_names=list(feature_names),
        retention_strategy=retention_strategy,
        input_summary=clean_audit_dict
    )

def cleanup_temp_files(*paths):
    for path in paths:
        try:
            if path and os.path.exists(path):
                os.remove(path)
        except Exception:
            pass

@app.post("/predict-batch")
async def predict_batch(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    output_format: Optional[str] = Query(None, description="Output format: csv, parquet, json, jsonl. Defaults to input format."),
    return_mode: str = Query("auto", description="'records' (returns JSON list), 'file' (returns scored file stream), 'summary' (stats + preview), 'auto'")
):
    # Ensure models are loaded
    if "xgb_model" not in models or "preprocessor" not in models:
        if os.path.exists(XGB_MODEL_PATH) and os.path.exists(PREPROCESSOR_PATH):
            models["preprocessor"] = joblib.load(PREPROCESSOR_PATH)
            models["xgb_model"] = joblib.load(XGB_MODEL_PATH)
            models["metadata"] = joblib.load(METADATA_PATH)
            if "explainer" not in models:
                models["explainer"] = shap.TreeExplainer(models["xgb_model"])
        else:
            raise HTTPException(status_code=503, detail="Models are not trained yet. Run model_engine.py first.")

    preprocessor = models["preprocessor"]
    xgb_model = models["xgb_model"]

    input_format = detect_file_format(file.filename or "data.csv")
    out_fmt = output_format.lower() if output_format else input_format
    if out_fmt not in ["csv", "parquet", "json", "jsonl", "feather"]:
        out_fmt = "csv"

    # Save uploaded stream to temporary file on disk using buffered streaming
    temp_dir = tempfile.gettempdir()
    in_temp = os.path.join(temp_dir, f"upload_{os.urandom(8).hex()}_{file.filename or 'data'}")
    out_temp = os.path.join(temp_dir, f"scored_{os.urandom(8).hex()}.{out_fmt}")

    try:
        with open(in_temp, "wb") as f_out:
            while chunk_data := await file.read(1024 * 1024 * 2):  # 2MB chunks
                f_out.write(chunk_data)
    except Exception as e:
        cleanup_temp_files(in_temp, out_temp)
        raise HTTPException(status_code=400, detail=f"Failed to receive uploaded file: {str(e)}")

    if os.path.getsize(in_temp) == 0:
        cleanup_temp_files(in_temp, out_temp)
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")

    # Process dataset in streaming chunks
    try:
        stats = process_large_dataset(
            input_file_path=in_temp,
            output_file_path=out_temp,
            input_format=input_format,
            output_format=out_fmt,
            preprocessor=preprocessor,
            xgb_model=xgb_model,
            chunk_size=50_000
        )
    except Exception as e:
        cleanup_temp_files(in_temp, out_temp)
        raise HTTPException(status_code=400, detail=f"Batch processing error: {str(e)}")

    # Remove input temp file
    cleanup_temp_files(in_temp)

    # Persist scored records into DuckDB analytical audit log
    if stats.get("preview_records"):
        try:
            background_tasks.add_task(
                save_predictions_audit,
                pd.DataFrame(stats["preview_records"]),
                f"batch_{os.path.splitext(file.filename or 'upload')[0]}"
            )
        except Exception as e:
            print(f"DuckDB batch audit warning: {e}")

    media_types = {
        "csv": "text/csv",
        "parquet": "application/vnd.apache.parquet",
        "json": "application/json",
        "jsonl": "application/x-ndjson",
        "feather": "application/octet-stream"
    }

    # 1. Summary mode
    if return_mode == "summary":
        background_tasks.add_task(cleanup_temp_files, out_temp)
        return {
            "status": "success",
            "total_rows": stats["total_rows"],
            "total_columns": stats["total_columns"],
            "mean_risk_score": stats["mean_risk_score"],
            "high_risk_count": stats["high_risk_count"],
            "moderate_risk_count": stats["moderate_risk_count"],
            "low_risk_count": stats["low_risk_count"],
            "preview_records": stats["preview_records"],
            "output_format": out_fmt
        }

    # 2. File / Stream mode or large datasets (> 10,000 rows in auto mode)
    if return_mode == "file" or (return_mode == "auto" and stats["total_rows"] > 10_000):
        background_tasks.add_task(cleanup_temp_files, out_temp)
        scored_filename = f"scored_{os.path.splitext(file.filename or 'data')[0]}.{out_fmt}"
        return FileResponse(
            path=out_temp,
            filename=scored_filename,
            media_type=media_types.get(out_fmt, "application/octet-stream"),
            headers={
                "X-Total-Rows": str(stats["total_rows"]),
                "X-Total-Columns": str(stats["total_columns"]),
                "X-Mean-Risk-Score": str(stats["mean_risk_score"]),
                "X-High-Risk-Count": str(stats["high_risk_count"]),
                "X-Low-Risk-Count": str(stats.get("low_risk_count", 0)),
                "X-Moderate-Risk-Count": str(stats.get("moderate_risk_count", 0)),
                "Access-Control-Expose-Headers": "X-Total-Rows, X-Total-Columns, X-Mean-Risk-Score, X-High-Risk-Count, X-Low-Risk-Count, X-Moderate-Risk-Count"
            }
        )

    # 3. Records mode (returns list of JSON records for smaller files)
    try:
        if out_fmt == "csv":
            df_res = pd.read_csv(out_temp)
        elif out_fmt == "parquet":
            df_res = pd.read_parquet(out_temp)
        elif out_fmt in ["json", "jsonl"]:
            df_res = pd.read_json(out_temp, lines=True)
        else:
            df_res = pd.read_csv(out_temp)

        result_records = df_res.to_dict(orient="records")
    finally:
        cleanup_temp_files(out_temp)

    return result_records

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend:app", host="0.0.0.0", port=8000, reload=True)
