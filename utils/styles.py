"""
utils/styles.py
───────────────
All custom CSS injected into the Streamlit app.
Call inject_css() once at app startup.
"""

import streamlit as st


CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

* { font-family: 'Inter', sans-serif; }

/* ── Background ── */
.stApp {
    background: linear-gradient(135deg, #0a0e1a 0%, #0d1b2e 40%, #0a1628 100%);
    color: #e2e8f0;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1b2e 0%, #101c30 100%);
    border-right: 1px solid rgba(99,179,237,0.15);
}

/* ── Hero header ── */
.main-header {
    background: linear-gradient(135deg, rgba(99,179,237,0.12) 0%, rgba(154,117,234,0.12) 100%);
    border: 1px solid rgba(99,179,237,0.25);
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    margin-bottom: 2rem;
}
.main-header h1 {
    font-size: 2.4rem;
    font-weight: 700;
    background: linear-gradient(135deg, #63b3ed, #9a75ea);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.4rem;
}
.main-header p { color: #94a3b8; font-size: 1rem; margin: 0; }

/* ── KPI metric cards ── */
.metric-card {
    background: linear-gradient(135deg, rgba(99,179,237,0.08), rgba(154,117,234,0.08));
    border: 1px solid rgba(99,179,237,0.2);
    border-radius: 12px;
    padding: 1.2rem;
    text-align: center;
    transition: all 0.3s ease;
}
.metric-card:hover { border-color: rgba(99,179,237,0.5); transform: translateY(-2px); }
.metric-card .metric-value { font-size: 2rem; font-weight: 700; color: #63b3ed; }
.metric-card .metric-label {
    font-size: 0.8rem; color: #94a3b8;
    text-transform: uppercase; letter-spacing: 0.05em;
}

/* ── Risk badges ── */
.risk-badge-high {
    background: linear-gradient(135deg, #fc8181, #f56565);
    color: white; border-radius: 8px; padding: 0.5rem 1rem;
    font-weight: 600; display: inline-block;
}
.risk-badge-low {
    background: linear-gradient(135deg, #68d391, #48bb78);
    color: white; border-radius: 8px; padding: 0.5rem 1rem;
    font-weight: 600; display: inline-block;
}

/* ── Section title ── */
.section-title {
    font-size: 1.3rem; font-weight: 600; color: #63b3ed;
    border-bottom: 2px solid rgba(99,179,237,0.3);
    padding-bottom: 0.5rem; margin-bottom: 1rem;
}

/* ── Boxes ── */
.info-box {
    background: rgba(99,179,237,0.08);
    border-left: 4px solid #63b3ed;
    border-radius: 0 8px 8px 0;
    padding: 1rem; margin: 0.5rem 0;
}
.warning-box {
    background: rgba(245,101,101,0.1);
    border-left: 4px solid #f56565;
    border-radius: 0 8px 8px 0;
    padding: 1rem; margin: 0.5rem 0;
}

/* ── Prediction result card ── */
.prediction-result {
    background: linear-gradient(135deg, rgba(99,179,237,0.12), rgba(154,117,234,0.12));
    border: 2px solid rgba(99,179,237,0.35);
    border-radius: 16px; padding: 2rem; margin: 1rem 0;
    text-align: center;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(99,179,237,0.06);
    border-radius: 10px; padding: 4px;
}
.stTabs [data-baseweb="tab"] { color: #94a3b8; border-radius: 8px; }
.stTabs [aria-selected="true"] {
    background: rgba(99,179,237,0.2) !important;
    color: #63b3ed !important;
}

/* ── Primary button ── */
button[kind="primary"] {
    background: linear-gradient(135deg, #63b3ed, #9a75ea) !important;
    border: none !important; border-radius: 8px !important;
    font-weight: 600 !important;
}

input[type="number"] { background: rgba(255,255,255,0.06) !important; }
</style>
"""


def inject_css() -> None:
    """Inject all custom CSS into the Streamlit page."""
    st.markdown(CSS, unsafe_allow_html=True)
