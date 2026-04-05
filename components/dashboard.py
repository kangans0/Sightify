"""
components/dashboard.py
────────────────────────
Renders the 📊 Dashboard page.

Shows model performance KPIs, confusion matrix, ROC curve,
feature importances, and dataset distribution charts.
"""

import streamlit as st
from utils.charts import (
    confusion_matrix_fig,
    distribution_fig,
    feature_importance_fig,
    roc_curve_fig,
    scatter_risk_fig,
)


def show_dashboard(df, metrics: dict, xgb_model) -> None:
    """
    Render the full Dashboard page.

    Parameters
    ──────────
    df        : preprocessed DataFrame (full dataset)
    metrics   : dict returned by train_models() — contains scores + raw predictions
    xgb_model : trained XGBClassifier (for feature importances)
    """
    st.markdown(
        '<div class="section-title">📊 Model Performance Overview</div>',
        unsafe_allow_html=True,
    )

    # ── KPI Cards ─────────────────────────────────────────────────────────
    kpis = [
        ("Accuracy",  f"{metrics['accuracy']:.2%}"),
        ("Precision", f"{metrics['precision']:.2%}"),
        ("Recall ★",  f"{metrics['recall']:.2%}"),
        ("F1-Score",  f"{metrics['f1']:.2%}"),
        ("ROC-AUC",   f"{metrics['roc_auc']:.3f}"),
    ]
    for col, (label, val) in zip(st.columns(5), kpis):
        with col:
            st.markdown(
                f"""<div class="metric-card">
                        <div class="metric-value">{val}</div>
                        <div class="metric-label">{label}</div>
                    </div>""",
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Confusion Matrix + ROC Curve (side by side) ────────────────────────
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(
            confusion_matrix_fig(metrics["y_test"], metrics["y_pred"]),
            use_container_width=True,
        )
    with col2:
        st.plotly_chart(
            roc_curve_fig(metrics["y_test"], metrics["y_proba"], metrics["roc_auc"]),
            use_container_width=True,
        )

    # ── Feature Importances ────────────────────────────────────────────────
    st.plotly_chart(
        feature_importance_fig(xgb_model, metrics["feature_names"]),
        use_container_width=True,
    )

    # ── Distribution + Scatter (side by side) ─────────────────────────────
    col3, col4 = st.columns(2)
    with col3:
        st.plotly_chart(
            distribution_fig(df, "health_risk_score", "Health Risk Score Distribution"),
            use_container_width=True,
        )
    with col4:
        st.plotly_chart(scatter_risk_fig(df), use_container_width=True)
