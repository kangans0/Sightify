"""
components/analytics.py
────────────────────────
Renders the 📈 Analytics page.

Three tabs:
  🔬 Distributions  — histogram + box plot for any chosen feature
  🔗 Correlations   — heatmap + interactive scatter
  📊 Group Analysis — grouped bars by obesity/age + violin plot
"""

import pandas as pd
import streamlit as st
from utils.charts import (
    box_plot_fig,
    correlation_heatmap,
    distribution_fig,
    grouped_bar_fig,
    scatter_pair_fig,
    violin_fig,
)


# Features available in the distribution dropdown
NUMERIC_FEATURES = [
    "health_risk_score", "sugar_percentage", "glucose_percentage",
    "cholesterol_percentage", "obesity_percentage",
    "heart_rate", "systolic", "diastolic", "age",
]


def show_analytics(df: pd.DataFrame) -> None:
    """
    Render the full Analytics page with three interactive tabs.

    Parameters
    ──────────
    df : preprocessed DataFrame (full 20,000-record dataset)
    """
    st.markdown(
        '<div class="section-title">📈 Dataset Analytics & Insights</div>',
        unsafe_allow_html=True,
    )

    tab1, tab2, tab3 = st.tabs(
        ["  🔬 Distributions  ", "  🔗 Correlations  ", "  📊 Group Analysis  "]
    )

    # ── Tab 1: Feature Distributions ──────────────────────────────────────
    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            feat = st.selectbox("Select Feature", NUMERIC_FEATURES)
        with c2:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(f"Distribution of **{feat}** by eye disease status")

        st.plotly_chart(
            distribution_fig(df, feat, f'{feat.replace("_", " ").title()} Distribution'),
            use_container_width=True,
        )
        st.plotly_chart(box_plot_fig(df, feat), use_container_width=True)

    # ── Tab 2: Correlations ────────────────────────────────────────────────
    with tab2:
        st.plotly_chart(correlation_heatmap(df), use_container_width=True)

        st.markdown("#### Scatter Plot")
        sc1, sc2 = st.columns(2)
        with sc1:
            x_feat = st.selectbox(
                "X axis",
                ["health_risk_score", "sugar_percentage", "glucose_percentage", "age"],
                key="x",
            )
        with sc2:
            y_feat = st.selectbox(
                "Y axis",
                ["cholesterol_percentage", "obesity_percentage", "systolic", "heart_rate"],
                key="y",
            )
        st.plotly_chart(scatter_pair_fig(df, x_feat, y_feat), use_container_width=True)

    # ── Tab 3: Group Analysis ──────────────────────────────────────────────
    with tab3:
        # Disease count by obesity group (reconstruct from one-hot columns)
        ob_cols = [c for c in df.columns if c.startswith("obesity_group_")]
        if ob_cols:
            ob_df = df.copy()
            ob_df["obesity_group"] = (
                ob_df[ob_cols].idxmax(axis=1).str.replace("obesity_group_", "")
            )
            st.plotly_chart(
                grouped_bar_fig(ob_df, "obesity_group", "Eye Disease by Obesity Group"),
                use_container_width=True,
            )

        # Disease count by age bucket
        df2 = df.copy()
        df2["age_bucket"] = pd.cut(
            df2["age"],
            bins=[0, 30, 45, 60, 100],
            labels=["< 30", "30-45", "45-60", "60+"],
        )
        st.plotly_chart(
            grouped_bar_fig(df2, "age_bucket", "Eye Disease by Age Group"),
            use_container_width=True,
        )

        # Violin plot: Health Risk Score
        st.plotly_chart(violin_fig(df), use_container_width=True)
