"""
components/about.py
────────────────────
Renders the ℹ️ About page.

Static informational content explaining:
  • XGBoost model setup and threshold choice
  • Linear Regression health risk model
  • Feature engineering overview
  • Medical disclaimer
"""

import streamlit as st


def show_about() -> None:
    """Render the static About page."""
    st.markdown(
        '<div class="section-title">ℹ️ About Sightify</div>',
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            """<div class="info-box">
            <h4 style="color:#63b3ed">🤖 XGBoost Classifier</h4>
            <p>Hyperparameter-tuned XGBoost model achieving <b>Recall of ~0.87</b>,
            optimised to minimise missed diagnoses (false negatives).</p>
            <ul>
              <li>Decision threshold: <b>0.30</b></li>
              <li>Max depth: 3 | Estimators: 200</li>
              <li>Learning rate: 0.1 | Subsample: 0.8</li>
            </ul>
            </div>""",
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """<div class="info-box">
            <h4 style="color:#9a75ea">📉 Health Risk Regression</h4>
            <p>A Linear Regression model trained on metabolic markers to produce
            a continuous <b>Health Risk Score (0–100)</b>.</p>
            <ul>
              <li>Inputs: sugar, glucose, cholesterol, obesity %</li>
              <li>Output: composite risk score</li>
              <li>Scaled with StandardScaler</li>
            </ul>
            </div>""",
            unsafe_allow_html=True,
        )

    # Feature Engineering card — separate call, no inline <br> between divs
    st.markdown(
        """<div style="margin-top:1rem" class="info-box">
        <h4 style="color:#63b3ed">📋 Feature Engineering</h4>
        <p>20 features including raw clinical measurements, computed metabolic
        risk indicators, and one-hot encoded categorical variables
        (obesity group, blood pressure category).</p>
        </div>""",
        unsafe_allow_html=True,
    )

    # Medical disclaimer — separate call keeps Streamlit's renderer happy
    st.markdown(
        """<div style="margin-top:1rem" class="warning-box">
        <b>⚠️ Medical Disclaimer</b><br>
        This tool is intended for <b>research and educational purposes only</b>.
        It is not a substitute for professional medical diagnosis or advice.
        Always consult a qualified healthcare provider for clinical decisions.
        </div>""",
        unsafe_allow_html=True,
    )
