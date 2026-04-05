"""
components/predict.py
──────────────────────
Renders the 🔮 Predict page.

Collects patient data via a form, runs:
  • Linear Regression  → Health Risk Score (gauge)
  • XGBoost Classifier → Disease probability + badge
Then displays result + radar chart + clinical recommendation.
"""

import numpy as np
import streamlit as st
from utils.charts import radar_chart, risk_gauge


def show_predict(metrics: dict, xgb_model, reg_model, scaler, risk_features: list) -> None:
    """
    Render the full Predict page.

    Parameters
    ──────────
    metrics       : dict from train_models() — contains threshold + feature_names
    xgb_model     : trained XGBClassifier
    reg_model     : trained LinearRegression (health risk score)
    scaler        : fitted StandardScaler for regression inputs
    risk_features : list of feature names the regression model uses
    """
    st.markdown(
        '<div class="section-title">🔮 Patient Risk Prediction</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        """<div class="info-box">
        Enter the patient's clinical data below. The XGBoost model will assess
        the probability of eye disease, and the regression model will compute
        a personalised <b>Health Risk Score</b>.
        </div>""",
        unsafe_allow_html=True,
    )
    st.markdown("<br>", unsafe_allow_html=True)

    # ── Input Form ─────────────────────────────────────────────────────────
    with st.form("patient_form"):
        st.markdown("#### 👤 Demographics & Metabolic Markers")
        c1, c2, c3 = st.columns(3)
        with c1:
            age          = st.number_input("Age (years)",        18,  100,  45)
            sugar        = st.number_input("Sugar %",           0.0, 100.0, 30.0, 0.5)
            glucose      = st.number_input("Glucose %",         0.0, 100.0, 25.0, 0.5)
        with c2:
            cholesterol  = st.number_input("Cholesterol %",     0.0, 100.0, 20.0, 0.5)
            obesity      = st.number_input("Obesity %",         0.0, 100.0, 15.0, 0.5)
            heart_rate   = st.number_input("Heart Rate (bpm)",   40,  200,  75)
        with c3:
            systolic     = st.number_input("Systolic BP (mmHg)", 80,  220, 120)
            diastolic    = st.number_input("Diastolic BP (mmHg)",40,  140,  80)
            metabolic_risk = st.number_input("Metabolic Risk Count", 0, 10, 2)

        st.markdown("#### 🩺 Clinical Flags & Categories")
        c4, c5 = st.columns(2)
        with c4:
            has_dr         = st.selectbox("Has Diabetic Retinopathy?", [0, 1],
                                          format_func=lambda x: "Yes" if x else "No")
            metabolic_flag = st.selectbox("Metabolic Syndrome Flag?",  [0, 1],
                                          format_func=lambda x: "Yes" if x else "No")
            obesity_group  = st.selectbox("Obesity Group",
                                          ["normal", "overweight", "obese", "underweight"])
        with c5:
            bp_category = st.selectbox("BP Category",
                                       ["normal", "elevated", "stage_1", "stage_2"])

        submitted = st.form_submit_button(
            "🔍 Run Prediction", type="primary", use_container_width=True
        )

    if not submitted:
        return

    # ── Step 1: Compute Health Risk Score (Linear Regression) ──────────────
    risk_lookup = {
        "sugar_percentage":        sugar,
        "cholesterol_percentage":  cholesterol,
        "glucose_percentage":      glucose,
        "obesity_percentage":      obesity,
    }
    risk_vals   = [risk_lookup.get(f, 0) for f in risk_features]
    risk_scaled = scaler.transform([risk_vals])
    health_risk_score = float(np.clip(reg_model.predict(risk_scaled)[0], 0, 100))

    # ── Step 2: Build full feature vector for XGBoost ──────────────────────
    feature_dict = {col: 0 for col in metrics["feature_names"]}
    feature_dict.update({
        "age":                    age,
        "sugar_percentage":       sugar,
        "glucose_percentage":     glucose,
        "cholesterol_percentage": cholesterol,
        "obesity_percentage":     obesity,
        "heart_rate":             heart_rate,
        "systolic":               systolic,
        "diastolic":              diastolic,
        "metabolic_risk_count":   metabolic_risk,
        "metabolic_syndrome_flag":metabolic_flag,
        "has_diabetic_retinopathy": has_dr,
        "health_risk_score":      health_risk_score,
    })
    # One-hot flags
    for key in [f"obesity_group_{obesity_group}", f"bp_category_{bp_category}"]:
        if key in feature_dict:
            feature_dict[key] = 1

    import pandas as pd
    input_df   = pd.DataFrame([feature_dict])[metrics["feature_names"]]
    proba      = xgb_model.predict_proba(input_df)[0][1]
    prediction = int(proba >= metrics["threshold"])

    # ── Step 3: Display Results ──────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("### 🧬 Prediction Results")

    r1, r2 = st.columns(2)
    with r1:
        badge = "risk-badge-high" if prediction == 1 else "risk-badge-low"
        label = "⚠️ Eye Disease Detected" if prediction == 1 else "✅ No Eye Disease Detected"
        prob_color = "#f56565" if prediction == 1 else "#68d391"
        st.markdown(
            f"""<div class="prediction-result">
                    <span class="{badge}">{label}</span><br><br>
                    <span style="font-size:2.2rem;font-weight:700;color:{prob_color}">
                        {proba:.1%}
                    </span><br>
                    <span style="color:#94a3b8;font-size:0.85rem">
                        Probability of Eye Disease
                    </span>
                </div>""",
            unsafe_allow_html=True,
        )
    with r2:
        st.plotly_chart(risk_gauge(health_risk_score), use_container_width=True)
        risk_label = (
            "🟢 Low Risk" if health_risk_score < 35
            else ("🟡 Moderate Risk" if health_risk_score < 60 else "🔴 High Risk")
        )
        st.markdown(
            f"<center style='color:#94a3b8'>{risk_label}</center>",
            unsafe_allow_html=True,
        )

    # ── Step 4: Clinical Recommendation ────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    if prediction == 1:
        st.markdown(
            """<div class="warning-box">
            <b>⚠️ Clinical Recommendation</b><br>
            The model has detected a high probability of eye disease. Please consult
            an ophthalmologist immediately. Diabetic retinopathy screening and a
            comprehensive eye examination are advised.
            </div>""",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """<div class="info-box">
            <b>✅ Clinical Recommendation</b><br>
            No immediate eye disease risk detected. Continue with regular annual
            eye check-ups and maintain healthy metabolic markers to reduce long-term risk.
            </div>""",
            unsafe_allow_html=True,
        )

    # ── Step 5: Radar Chart (Metabolic Profile) ─────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### 📡 Patient Metabolic Profile")
    cats = ["Sugar %", "Glucose %", "Cholesterol %", "Obesity %", "Heart Rate", "Health Risk"]
    vals = [
        sugar,
        glucose,
        cholesterol,
        obesity,
        (heart_rate - 40) / 160 * 100,
        health_risk_score,
    ]
    st.plotly_chart(radar_chart(vals, cats), use_container_width=True)
