"""
app.py
──────
EyeSight AI — Eye Disease Prediction System
Entry point for the Streamlit application.

Project structure
─────────────────
app.py                     ← You are here (page config + routing only)
models/
  trainer.py               ← load_data(), train_models()
utils/
  styles.py                ← inject_css()
  charts.py                ← all Plotly chart factories
components/
  dashboard.py             ← 📊 Dashboard page
  predict.py               ← 🔮 Predict page
  analytics.py             ← 📈 Analytics page
  about.py                 ← ℹ️  About page

Run
───
  streamlit run app.py
"""

import streamlit as st

# ── Page config (must be the very first Streamlit call) ───────────────────────
st.set_page_config(
    page_title="Sightify | Eye Disease Prediction",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Local imports ─────────────────────────────────────────────────────────────
from utils.styles import inject_css
from models.trainer import load_data, train_models
from components.dashboard import show_dashboard
from components.predict import show_predict
from components.analytics import show_analytics
from components.about import show_about


def main() -> None:
    # ── Inject global CSS ──────────────────────────────────────────────────
    inject_css()

    # ── Hero header ────────────────────────────────────────────────────────
    st.markdown(
        """
        <div class="main-header">
            <h1>👁️ Sightify</h1>
            <p>Advanced Eye Disease Prediction · XGBoost · Health Risk Analytics ·
            Powered by Machine Learning</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Load data + train models (cached after first run) ──────────────────
    with st.spinner("⚙️ Loading data and training models..."):
        df = load_data()
        xgb_model, metrics, reg_model, scaler, risk_features = train_models(df)

    if xgb_model is None:
        st.stop()

    # ── Sidebar ────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### 🔭 Navigation")
        page = st.radio(
            "",
            ["📊 Dashboard", "🔮 Predict", "📈 Analytics", "ℹ️ About"],
            label_visibility="collapsed",
        )

        st.markdown("---")
        st.markdown("### 🤖 Model Info")
        st.markdown(
            f"""<div class="info-box">
            <b>Algorithm:</b> XGBoost<br>
            <b>Recall:</b> {metrics['recall']:.2%}<br>
            <b>ROC-AUC:</b> {metrics['roc_auc']:.3f}<br>
            <b>Threshold:</b> {metrics['threshold']}<br>
            <b>Features:</b> {len(metrics['feature_names'])}
            </div>""",
            unsafe_allow_html=True,
        )

        st.markdown("---")
        n_pos = int(df["has_eye_disease"].sum())
        n_neg = len(df) - n_pos
        st.markdown(
            f"""<div class="info-box">
            <b>🗂️ Dataset</b><br>
            Total: {len(df):,} records<br>
            With Disease: {n_pos:,} ({n_pos/len(df):.1%})<br>
            Without: {n_neg:,} ({n_neg/len(df):.1%})
            </div>""",
            unsafe_allow_html=True,
        )

    # ── Page Routing ───────────────────────────────────────────────────────
    if page == "📊 Dashboard":
        show_dashboard(df, metrics, xgb_model)

    elif page == "🔮 Predict":
        show_predict(metrics, xgb_model, reg_model, scaler, risk_features)

    elif page == "📈 Analytics":
        show_analytics(df)

    elif page == "ℹ️ About":
        show_about()


if __name__ == "__main__":
    main()
