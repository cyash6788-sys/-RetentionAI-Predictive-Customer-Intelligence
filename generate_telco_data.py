"""
Synthetic Telco Customer Churn Dataset Generator for Survival Analysis & ML
Generates 100,000 rows with realistic hazard models, censoring, and feature relationships.
"""

import numpy as np
import pandas as pd

def generate_telco_churn_dataset(n_samples: int = 100_000, random_seed: int = 42) -> pd.DataFrame:
    np.random.seed(random_seed)

    # 1. Demographics & Account Info
    customer_ids = [f"CUST-{100000 + i}" for i in range(n_samples)]
    gender = np.random.choice(["Male", "Female"], size=n_samples, p=[0.5, 0.5])
    senior_citizen = np.random.choice([0, 1], size=n_samples, p=[0.84, 0.16])
    partner = np.random.choice(["Yes", "No"], size=n_samples, p=[0.48, 0.52])
    dependents = np.where(partner == "Yes",
                          np.random.choice(["Yes", "No"], size=n_samples, p=[0.50, 0.50]),
                          np.random.choice(["Yes", "No"], size=n_samples, p=[0.12, 0.88]))

    # 2. Services
    phone_service = np.random.choice(["Yes", "No"], size=n_samples, p=[0.90, 0.10])
    multiple_lines = np.where(phone_service == "Yes",
                              np.random.choice(["Yes", "No"], size=n_samples, p=[0.45, 0.55]),
                              "No phone service")

    internet_service = np.random.choice(["DSL", "Fiber optic", "No"], size=n_samples, p=[0.35, 0.44, 0.21])

    online_security = []
    online_backup = []
    device_protection = []
    tech_support = []
    streaming_tv = []
    streaming_movies = []

    for inet in internet_service:
        if inet == "No":
            online_security.append("No internet service")
            online_backup.append("No internet service")
            device_protection.append("No internet service")
            tech_support.append("No internet service")
            streaming_tv.append("No internet service")
            streaming_movies.append("No internet service")
        else:
            online_security.append(np.random.choice(["Yes", "No"], p=[0.38, 0.62]))
            online_backup.append(np.random.choice(["Yes", "No"], p=[0.42, 0.58]))
            device_protection.append(np.random.choice(["Yes", "No"], p=[0.41, 0.59]))
            tech_support.append(np.random.choice(["Yes", "No"], p=[0.36, 0.64]))
            streaming_tv.append(np.random.choice(["Yes", "No"], p=[0.49, 0.51]))
            streaming_movies.append(np.random.choice(["Yes", "No"], p=[0.50, 0.50]))

    # 3. Contract & Billing
    contract_type = np.random.choice(["Month-to-month", "One year", "Two year"], size=n_samples, p=[0.55, 0.24, 0.21])
    paperless_billing = np.random.choice(["Yes", "No"], size=n_samples, p=[0.59, 0.41])
    payment_method = np.random.choice([
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ], size=n_samples, p=[0.34, 0.23, 0.22, 0.21])

    # 4. Monthly Charges based on services
    base_charge = np.where(phone_service == "Yes", 20.0, 10.0)
    inet_charge = np.select(
        [internet_service == "Fiber optic", internet_service == "DSL", internet_service == "No"],
        [45.0, 25.0, 0.0]
    )
    addons = (
        (np.array(multiple_lines) == "Yes") * 8.0 +
        (np.array(online_security) == "Yes") * 7.0 +
        (np.array(online_backup) == "Yes") * 6.0 +
        (np.array(device_protection) == "Yes") * 6.0 +
        (np.array(tech_support) == "Yes") * 8.0 +
        (np.array(streaming_tv) == "Yes") * 10.0 +
        (np.array(streaming_movies) == "Yes") * 10.0
    )
    noise = np.random.normal(0, 2.5, size=n_samples)
    monthly_charges = np.round(np.clip(base_charge + inet_charge + addons + noise, 18.25, 118.75), 2)

    # 5. Survival Analysis Event Generation (Weibull/Cox Proportional Hazards Model)
    # Log-hazard components:
    # High churn risk: Month-to-month, Fiber optic, Electronic check, High monthly charges, No tech support/security
    # Low churn risk: Two year contract, One year contract, Partner, Dependents, Automatic payment
    beta_contract = np.select(
        [contract_type == "Month-to-month", contract_type == "One year", contract_type == "Two year"],
        [1.3, -0.6, -1.8]
    )
    beta_internet = np.select(
        [internet_service == "Fiber optic", internet_service == "DSL", internet_service == "No"],
        [0.45, 0.0, -0.7]
    )
    beta_tech_support = np.where(np.array(tech_support) == "Yes", -0.4, 0.15)
    beta_online_sec = np.where(np.array(online_security) == "Yes", -0.35, 0.15)
    beta_payment = np.where(payment_method == "Electronic check", 0.4, -0.15)
    beta_senior = np.where(senior_citizen == 1, 0.2, -0.05)
    beta_charges = (monthly_charges - 65.0) / 30.0 * 0.35
    beta_partner = np.where(partner == "Yes", -0.2, 0.1)

    log_hazard = (
        -3.2  # baseline intercept
        + beta_contract
        + beta_internet
        + beta_tech_support
        + beta_online_sec
        + beta_payment
        + beta_senior
        + beta_charges
        + beta_partner
    )

    hazard_ratio = np.exp(log_hazard)

    # Inversion method for Weibull event time:
    # S(t) = exp(-lambda * t^alpha * hazard_ratio) => T = (-ln(U) / (lambda * hazard_ratio))^(1/alpha)
    weibull_alpha = 1.15  # slightly increasing or constant hazard
    weibull_lambda = 0.015
    u = np.random.uniform(1e-6, 1.0, size=n_samples)
    event_times = (-np.log(u) / (weibull_lambda * hazard_ratio)) ** (1.0 / weibull_alpha)

    # Censoring time (observation window / customer lifetime up to 72 months)
    # Most customers have censoring times distributed between 1 and 72 months
    censor_times = np.random.exponential(scale=65.0, size=n_samples) + np.random.uniform(5, 72, size=n_samples)
    censor_times = np.clip(censor_times, 1.0, 72.0)

    # Observed tenure and event indicator (Churn)
    observed_time = np.minimum(event_times, censor_times)
    tenure = np.clip(np.ceil(observed_time).astype(int), 1, 72)
    churn = (event_times <= censor_times).astype(int)

    # 6. Total Charges calculated as monthly * tenure with minor variability
    total_charges = np.round(np.clip(monthly_charges * tenure * np.random.uniform(0.96, 1.04, size=n_samples), 18.0, 8900.0), 2)

    df = pd.DataFrame({
        "customerID": customer_ids,
        "gender": gender,
        "SeniorCitizen": senior_citizen,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract_type,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
        "Churn": churn
    })

    return df

if __name__ == "__main__":
    print("Generating 100,000 synthetic telco records for survival & churn analysis...")
    df = generate_telco_churn_dataset(n_samples=100_000, random_seed=42)
    churn_rate = df["Churn"].mean() * 100
    print(f"Dataset generated. Shape: {df.shape}")
    print(f"Overall Churn rate: {churn_rate:.2f}% (Censored: {100 - churn_rate:.2f}%)")
    print(f"Median tenure: {df['tenure'].median()} months (Min: {df['tenure'].min()}, Max: {df['tenure'].max()})")
    df.to_csv("telco_data.csv", index=False)
    print("Successfully saved to telco_data.csv")
