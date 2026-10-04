"""
Model Engine: Survival Analysis (Kaplan-Meier) & XGBoost Churn Classifier
- Fits Kaplan-Meier survival curves using lifelines
- Builds Scikit-Learn ColumnTransformer preprocessing pipeline
- Trains XGBClassifier with scale_pos_weight for class imbalance
- Saves artifacts using joblib for FastAPI backend & Streamlit frontend
"""

import os
import joblib
import pandas as pd
import numpy as np
from lifelines import KaplanMeierFitter
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import classification_report, roc_auc_score, brier_score_loss
from xgboost import XGBClassifier

def train_and_save_models(data_path: str = "telco_data.csv", output_dir: str = "artifacts"):
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} records.")
    
    # ----------------------------------------------------
    # 1. Macro Survival Analysis: Kaplan-Meier Estimation
    # ----------------------------------------------------
    print("Fitting Kaplan-Meier Survival Curves...")
    kmf_overall = KaplanMeierFitter()
    kmf_overall.fit(durations=df["tenure"], event_observed=df["Churn"], label="Overall Customer Base")
    
    # Stratified KM curves for deeper macro insights
    km_by_contract = {}
    for contract in df["Contract"].unique():
        kmf = KaplanMeierFitter()
        mask = df["Contract"] == contract
        kmf.fit(durations=df.loc[mask, "tenure"], event_observed=df.loc[mask, "Churn"], label=contract)
        km_by_contract[contract] = {
            "timeline": kmf.survival_function_.index.tolist(),
            "survival_prob": kmf.survival_function_[contract].tolist(),
            "ci_lower": kmf.confidence_interval_survival_function_.iloc[:, 0].tolist(),
            "ci_upper": kmf.confidence_interval_survival_function_.iloc[:, 1].tolist(),
            "median_survival_time": float(kmf.median_survival_time_) if not np.isnan(kmf.median_survival_time_) else None
        }

    km_by_internet = {}
    for inet in df["InternetService"].unique():
        kmf = KaplanMeierFitter()
        mask = df["InternetService"] == inet
        kmf.fit(durations=df.loc[mask, "tenure"], event_observed=df.loc[mask, "Churn"], label=inet)
        km_by_internet[inet] = {
            "timeline": kmf.survival_function_.index.tolist(),
            "survival_prob": kmf.survival_function_[inet].tolist(),
            "median_survival_time": float(kmf.median_survival_time_) if not np.isnan(kmf.median_survival_time_) else None
        }

    km_data = {
        "overall": {
            "timeline": kmf_overall.survival_function_.index.tolist(),
            "survival_prob": kmf_overall.survival_function_["Overall Customer Base"].tolist(),
            "ci_lower": kmf_overall.confidence_interval_survival_function_.iloc[:, 0].tolist(),
            "ci_upper": kmf_overall.confidence_interval_survival_function_.iloc[:, 1].tolist(),
            "median_survival_time": float(kmf_overall.median_survival_time_) if not np.isnan(kmf_overall.median_survival_time_) else None
        },
        "by_contract": km_by_contract,
        "by_internet": km_by_internet,
        "event_table": kmf_overall.event_table.reset_index().to_dict(orient="records")
    }

    joblib.dump(kmf_overall, os.path.join(output_dir, "kmf_model.joblib"))
    joblib.dump(km_data, os.path.join(output_dir, "km_data.joblib"))
    print(f"Kaplan-Meier overall median survival time: {kmf_overall.median_survival_time_} months")

    # ----------------------------------------------------
    # 2. Scikit-Learn Preprocessing Pipeline
    # ----------------------------------------------------
    print("Building Scikit-Learn Preprocessing Pipeline...")
    numeric_features = ["tenure", "MonthlyCharges", "TotalCharges"]
    categorical_features = [
        "gender",
        "SeniorCitizen",
        "Partner",
        "Dependents",
        "PhoneService",
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "Contract",
        "PaperlessBilling",
        "PaymentMethod"
    ]

    feature_cols = numeric_features + categorical_features
    X = df[feature_cols].copy()
    y = df["Churn"].values

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features)
        ],
        remainder="drop"
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Fitting preprocessor on {len(X_train)} train samples...")
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    # Extract transformed feature names
    cat_encoder = preprocessor.named_transformers_["cat"]
    transformed_cat_names = cat_encoder.get_feature_names_out(categorical_features).tolist()
    all_transformed_feature_names = numeric_features + transformed_cat_names

    # ----------------------------------------------------
    # 3. XGBoost Classifier with scale_pos_weight
    # ----------------------------------------------------
    num_neg = np.sum(y_train == 0)
    num_pos = np.sum(y_train == 1)
    scale_pos_weight = num_neg / num_pos
    print(f"Class distribution - Negative: {num_neg}, Positive: {num_pos}, scale_pos_weight: {scale_pos_weight:.3f}")

    print("Training XGBClassifier...")
    xgb_model = XGBClassifier(
        n_estimators=180,
        max_depth=5,
        learning_rate=0.08,
        subsample=0.85,
        colsample_bytree=0.85,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1
    )

    xgb_model.fit(X_train_transformed, y_train)

    # Evaluation
    y_pred = xgb_model.predict(X_test_transformed)
    y_pred_proba = xgb_model.predict_proba(X_test_transformed)[:, 1]

    roc_auc = roc_auc_score(y_test, y_pred_proba)
    brier = brier_score_loss(y_test, y_pred_proba)
    print("\n--- Test Set Evaluation ---")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"Brier Score: {brier:.4f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    # ----------------------------------------------------
    # 4. Save Artifacts with Joblib
    # ----------------------------------------------------
    joblib.dump(preprocessor, os.path.join(output_dir, "preprocessor.joblib"))
    joblib.dump(xgb_model, os.path.join(output_dir, "xgb_model.joblib"))
    
    metadata = {
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "feature_cols": feature_cols,
        "transformed_feature_names": all_transformed_feature_names,
        "roc_auc": float(roc_auc),
        "scale_pos_weight": float(scale_pos_weight)
    }
    joblib.dump(metadata, os.path.join(output_dir, "metadata.joblib"))

    print(f"\nAll artifacts successfully saved to '{output_dir}/'")

if __name__ == "__main__":
    train_and_save_models()
