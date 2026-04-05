"""
utils/charts.py
───────────────
All Plotly chart factory functions used across the app.
Every function receives data and returns a Plotly Figure.
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix, roc_curve


# ── Shared layout applied to every chart ──────────────────────────────────────
PLOT_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter", color="#e2e8f0"),
    margin=dict(t=50, b=40, l=40, r=40),
    xaxis=dict(
        gridcolor="rgba(255,255,255,0.07)",
        zerolinecolor="rgba(255,255,255,0.12)",
    ),
    yaxis=dict(
        gridcolor="rgba(255,255,255,0.07)",
        zerolinecolor="rgba(255,255,255,0.12)",
    ),
)


# ── Dashboard charts ───────────────────────────────────────────────────────────

def confusion_matrix_fig(y_test, y_pred) -> go.Figure:
    """2×2 heatmap showing TP / TN / FP / FN counts."""
    cm = confusion_matrix(y_test, y_pred)
    labels = ["No Disease", "Eye Disease"]
    fig = px.imshow(
        cm, text_auto=True, x=labels, y=labels,
        color_continuous_scale="Blues",
        title="Confusion Matrix",
        labels=dict(x="Predicted", y="Actual"),
    )
    fig.update_traces(textfont_size=16)
    fig.update_layout(
        **PLOT_LAYOUT,
        coloraxis_showscale=False,
        title=dict(font_size=16, x=0.5),
    )
    return fig


def roc_curve_fig(y_test, y_proba, auc: float) -> go.Figure:
    """ROC curve with AUC score annotated in the legend."""
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr, mode="lines",
        name=f"ROC (AUC={auc:.3f})",
        line=dict(color="#63b3ed", width=2.5),
    ))
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines", name="Random",
        line=dict(color="#94a3b8", width=1.5, dash="dash"),
    ))
    fig.update_layout(
        **PLOT_LAYOUT,
        title="ROC Curve",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        title_x=0.5,
        legend=dict(x=0.65, y=0.1),
    )
    return fig


def feature_importance_fig(model, feature_names: list) -> go.Figure:
    """Horizontal bar chart of the top-15 XGBoost feature importances."""
    imp = pd.Series(model.feature_importances_, index=feature_names)
    imp = imp.sort_values(ascending=True).tail(15)
    fig = px.bar(
        x=imp.values, y=imp.index, orientation="h",
        title="Top 15 Feature Importances",
        color=imp.values,
        color_continuous_scale="Blues",
    )
    fig.update_layout(
        **PLOT_LAYOUT,
        coloraxis_showscale=False,
        xaxis_title="Importance",
        yaxis_title="",
        title_x=0.5,
    )
    return fig


def distribution_fig(df: pd.DataFrame, col: str, title: str) -> go.Figure:
    """Overlapping histogram split by eye-disease status."""
    fig = px.histogram(
        df, x=col, color="has_eye_disease",
        barmode="overlay", nbins=40,
        color_discrete_map={0: "#63b3ed", 1: "#f56565"},
        labels={"has_eye_disease": "Eye Disease"},
        title=title,
        opacity=0.75,
    )
    fig.update_layout(**PLOT_LAYOUT, title_x=0.5, bargap=0.05)
    return fig


def scatter_risk_fig(df: pd.DataFrame) -> go.Figure:
    """Scatter: Health Risk Score vs Sugar %, coloured by disease status."""
    fig = px.scatter(
        df, x="health_risk_score", y="sugar_percentage",
        color="has_eye_disease",
        color_discrete_map={0: "#63b3ed", 1: "#f56565"},
        opacity=0.55,
        labels={"has_eye_disease": "Eye Disease"},
        title="Health Risk Score vs Sugar %",
    )
    fig.update_layout(**PLOT_LAYOUT, title_x=0.5)
    return fig


# ── Predict page charts ────────────────────────────────────────────────────────

def risk_gauge(score: float, label: str = "Health Risk Score") -> go.Figure:
    """Speedometer-style gauge (0–100) colour-coded by risk level."""
    color = "#68d391" if score < 35 else ("#f6ad55" if score < 60 else "#f56565")
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        domain={"x": [0, 1], "y": [0, 1]},
        title={"text": label, "font": {"color": "#e2e8f0", "size": 14}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#94a3b8"},
            "bar": {"color": color, "thickness": 0.25},
            "bgcolor": "rgba(255,255,255,0.04)",
            "borderwidth": 0,
            "threshold": {
                "line": {"color": "#f56565", "width": 3},
                "thickness": 0.75,
                "value": 70,
            },
            "steps": [
                {"range": [0, 35],  "color": "rgba(104,211,145,0.15)"},
                {"range": [35, 60], "color": "rgba(246,173,85,0.15)"},
                {"range": [60, 100],"color": "rgba(245,101,101,0.15)"},
            ],
        },
        number={"font": {"color": color, "size": 28}},
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        font={"family": "Inter", "color": "#e2e8f0"},
        margin=dict(t=40, b=10, l=20, r=20),
        height=200,
    )
    return fig


def radar_chart(radar_vals: list, radar_cats: list) -> go.Figure:
    """Polar/radar chart showing patient's metabolic profile across 6 axes."""
    fig = go.Figure(go.Scatterpolar(
        r=radar_vals + [radar_vals[0]],
        theta=radar_cats + [radar_cats[0]],
        fill="toself",
        fillcolor="rgba(99,179,237,0.2)",
        line=dict(color="#63b3ed", width=2),
        name="Patient",
    ))
    fig.update_layout(
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(
                visible=True, range=[0, 100],
                gridcolor="rgba(255,255,255,0.1)",
                tickfont=dict(color="#94a3b8"),
            ),
            angularaxis=dict(
                gridcolor="rgba(255,255,255,0.1)",
                tickcolor="#e2e8f0",
                tickfont=dict(color="#e2e8f0"),
            ),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e2e8f0", family="Inter"),
        margin=dict(t=30, b=30, l=40, r=40),
        showlegend=False,
        height=320,
    )
    return fig


# ── Analytics page charts ──────────────────────────────────────────────────────

def box_plot_fig(df: pd.DataFrame, col: str) -> go.Figure:
    """Box plot of a feature split by disease status."""
    fig = px.box(
        df, y=col, color="has_eye_disease",
        color_discrete_map={0: "#63b3ed", 1: "#f56565"},
        labels={"has_eye_disease": "Eye Disease (0=No, 1=Yes)"},
        title=f'{col.replace("_", " ").title()} — Box Plot by Disease Status',
    )
    fig.update_layout(**PLOT_LAYOUT, title_x=0.5)
    return fig


def correlation_heatmap(df: pd.DataFrame) -> go.Figure:
    """Pearson correlation heatmap for the top 12 numeric features."""
    num_cols = df.select_dtypes(include=[np.number]).columns[:12]
    corr = df[num_cols].corr()
    fig = px.imshow(
        corr, text_auto=".2f",
        color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1,
        title="Feature Correlation Heatmap",
    )
    fig.update_layout(**PLOT_LAYOUT, title_x=0.5, width=None)
    fig.update_traces(textfont_size=9)
    return fig


def scatter_pair_fig(df: pd.DataFrame, x_feat: str, y_feat: str) -> go.Figure:
    """Scatter plot for any pair of features, coloured by disease status."""
    fig = px.scatter(
        df, x=x_feat, y=y_feat,
        color="has_eye_disease",
        color_discrete_map={0: "#63b3ed", 1: "#f56565"},
        opacity=0.5,
        title=f"{x_feat} vs {y_feat}",
        labels={"has_eye_disease": "Eye Disease"},
    )
    fig.update_layout(**PLOT_LAYOUT, title_x=0.5)
    return fig


def grouped_bar_fig(df: pd.DataFrame, group_col: str, title: str) -> go.Figure:
    """Grouped bar chart: disease count by a categorical group column."""
    grp = (
        df.groupby([group_col, "has_eye_disease"])
        .size()
        .reset_index(name="count")
    )
    fig = px.bar(
        grp, x=group_col, y="count", color="has_eye_disease",
        color_discrete_map={0: "#63b3ed", 1: "#f56565"},
        barmode="group",
        title=title,
        labels={
            "has_eye_disease": "Eye Disease",
            "count": "Patients",
            group_col: group_col.replace("_", " ").title(),
        },
    )
    fig.update_layout(**PLOT_LAYOUT, title_x=0.5)
    return fig


def violin_fig(df: pd.DataFrame) -> go.Figure:
    """Violin + box plot of Health Risk Score by disease class."""
    fig = px.violin(
        df, y="health_risk_score", x="has_eye_disease",
        box=True, points="outliers",
        color="has_eye_disease",
        color_discrete_map={0: "#63b3ed", 1: "#f56565"},
        title="Health Risk Score Distribution by Disease Status",
        labels={
            "has_eye_disease": "Eye Disease",
            "health_risk_score": "Health Risk Score",
        },
    )
    fig.update_layout(**PLOT_LAYOUT, title_x=0.5)
    return fig
