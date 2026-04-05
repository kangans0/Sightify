"""
models/trainer.py
─────────────────
Handles all data loading and model training.

Public API
──────────
    load_data()    → pd.DataFrame
    train_models() → (xgb_model, metrics_dict, reg_model, scaler, risk_features)
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import streamlit as st
import xgboost as xgb
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ── Data Loading ───────────────────────────────────────────────────────────────

@st.cache_data
def load_data() -> pd.DataFrame:
    """
    Load and preprocess processed_dataset.csv.

    Steps:
      1. Drop metadata columns (name, Unnamed, blood_pressure)
      2. Cast bool columns to int  (0/1)
      3. One-hot encode obesity_group and bp_category
      4. Fallback: compute health_risk_score if missing
    """
    df = pd.read_csv("processed_dataset.csv")

    # 1. Drop unused metadata columns
    drop_cols = [c for c in ["Unnamed: 0", "name", "blood_pressure"] if c in df.columns]
    df.drop(columns=drop_cols, inplace=True)

    # 2. Booleans → int (XGBoost requires numeric input)
    for col in df.select_dtypes(include=["bool"]).columns:
        df[col] = df[col].astype(int)

    # 3. One-hot encode categorical columns
    if "obesity_group" in df.columns:
        df = pd.get_dummies(df, columns=["obesity_group"], prefix="obesity_group")
    if "bp_category" in df.columns:
        df = pd.get_dummies(df, columns=["bp_category"], prefix="bp_category")

    # 4. Safety fallback for health_risk_score
    if "health_risk_score" not in df.columns:
        df["health_risk_score"] = (
            df.get("sugar_percentage", 0) * 0.3
            + df.get("cholesterol_percentage", 0) * 0.3
            + df.get("glucose_percentage", 0) * 0.2
            + df.get("obesity_percentage", 0) * 0.2
        )

    return df


# ── Model Training ─────────────────────────────────────────────────────────────

@st.cache_resource
def train_models(df: pd.DataFrame):
    """
    Train the XGBoost classifier and the Linear Regression risk scorer.

    XGBoost
    ───────
    Hyperparameters selected via GridSearchCV in the development notebook.
    Decision threshold is set to 0.30 (instead of the default 0.50) to
    maximise Recall — the clinical priority is to catch as many sick
    patients as possible, even at the cost of some false positives.

    Linear Regression (Health Risk Score)
    ─────────────────────────────────────
    Trained on four metabolic % features to produce a composite score
    in the range [0, 100].

    Returns
    ───────
    xgb_model     : trained XGBClassifier
    metrics       : dict of evaluation metrics + raw predictions
    reg_model     : trained LinearRegression for risk scoring
    scaler        : fitted StandardScaler (for inference)
    risk_features : list of feature names used by the regression model
    """
    TARGET = "has_eye_disease"

    if TARGET not in df.columns:
        st.error("Target column 'has_eye_disease' not found in dataset.")
        return None, None, None, None, None

    feature_cols = [c for c in df.columns if c != TARGET]
    X, y = df[feature_cols], df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ── XGBoost Classifier ────────────────────────────────────────────────
    xgb_model = xgb.XGBClassifier(
        colsample_bytree=0.8,
        gamma=0,
        learning_rate=0.1,
        max_depth=3,
        min_child_weight=3,
        n_estimators=200,
        subsample=0.8,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
        tree_method="hist",
    )
    xgb_model.fit(X_train, y_train)

    THRESHOLD = 0.30  # Tuned for Recall ≈ 0.87
    y_proba = xgb_model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= THRESHOLD).astype(int)

    metrics = {
        "accuracy":      accuracy_score(y_test, y_pred),
        "precision":     precision_score(y_test, y_pred, zero_division=0),
        "recall":        recall_score(y_test, y_pred, zero_division=0),
        "f1":            f1_score(y_test, y_pred, zero_division=0),
        "roc_auc":       roc_auc_score(y_test, y_proba),
        "y_test":        y_test,
        "y_pred":        y_pred,
        "y_proba":       y_proba,
        "threshold":     THRESHOLD,
        "feature_names": list(X.columns),
    }

    # ── Linear Regression – Health Risk Score ─────────────────────────────
    RISK_FEATURES = [
        "sugar_percentage", "cholesterol_percentage",
        "glucose_percentage", "obesity_percentage",
    ]
    risk_features = [c for c in RISK_FEATURES if c in df.columns]

    scaler = StandardScaler()
    X_risk = scaler.fit_transform(df[risk_features])
    reg_model = LinearRegression().fit(X_risk, df["health_risk_score"])

    return xgb_model, metrics, reg_model, scaler, risk_features
