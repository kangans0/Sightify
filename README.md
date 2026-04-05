# 👁️ Sightify — Eye Disease Prediction System

> An interactive, ML-powered Streamlit dashboard that predicts the risk of eye disease using a hypertuned XGBoost classifier and computes a personalised Health Risk Score via a Linear Regression model.

---

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the app
streamlit run app.py
```

The app will open at **http://localhost:8501**

---

## 📁 Project Structure

```
final/
├── app.py                  ← Main Streamlit application
├── processed_dataset.csv   ← Cleaned & feature-engineered dataset (20,000 records)
├── requirements.txt        ← Pinned Python dependencies
├── EyeDisease (3).ipynb    ← Source notebook (model development & EDA)
└── README.md               ← This file
```

---

## 🧠 Machine Learning Architecture

The system uses **two models** working together:

### Model 1 — XGBoost Classifier (Primary)
> **Purpose:** Predict whether a patient has eye disease (binary classification)

| Parameter | Value |
|---|---|
| Algorithm | `XGBClassifier` |
| `learning_rate` | 0.1 |
| `max_depth` | 3 |
| `n_estimators` | 200 |
| `subsample` | 0.8 |
| `colsample_bytree` | 0.8 |
| `min_child_weight` | 3 |
| `gamma` | 0 |
| Decision Threshold | **0.30** (tuned to maximise Recall) |
| Target Metric | **Recall ≈ 0.87** (minimise missed diagnoses) |

**Why threshold 0.30?**  
In medical AI, it is worse to *miss* a patient who has a disease (false negative) than to flag a healthy patient for further screening (false positive). The threshold is deliberately lowered from the default 0.50 to 0.30, which pushes Recall from ~0.50 up to **~0.86–0.87**, ensuring the model catches the vast majority of at-risk patients.

---

### Model 2 — Linear Regression (Health Risk Score)
> **Purpose:** Produce a continuous score (0–100) representing a patient's overall metabolic risk

**Inputs:**
- `sugar_percentage`
- `cholesterol_percentage`
- `glucose_percentage`
- `obesity_percentage`

These four features are standardised via `StandardScaler` before being passed to the regression model.

**Output:** A single float score clamped to `[0, 100]`:
- 🟢 **0–34** → Low Risk
- 🟡 **35–59** → Moderate Risk
- 🔴 **60–100** → High Risk

---

## 📊 Dataset Overview

| Property | Value |
|---|---|
| Source | `processed_dataset.csv` |
| Records | 20,000 patients |
| Target | `has_eye_disease` (0 = No, 1 = Yes) |
| Class balance | ~49% disease / 51% no disease |
| Train / Test split | 80% / 20% |

### Features Used (20 total)

| Feature | Type | Description |
|---|---|---|
| `age` | Numeric | Patient age in years |
| `has_diabetic_retinopathy` | Binary (0/1) | Whether patient has DR (top predictor) |
| `sugar_percentage` | Numeric | Blood sugar level as % |
| `glucose_percentage` | Numeric | Glucose reading as % |
| `cholesterol_percentage` | Numeric | Cholesterol as % |
| `obesity_percentage` | Numeric | Obesity indicator as % |
| `heart_rate` | Numeric | Heart rate in bpm |
| `systolic` | Numeric | Systolic blood pressure (mmHg) |
| `diastolic` | Numeric | Diastolic blood pressure (mmHg) |
| `metabolic_risk_count` | Numeric | Count of active metabolic risk factors |
| `metabolic_syndrome_flag` | Binary (0/1) | Whether patient meets metabolic syndrome criteria |
| `health_risk_score` | Numeric | Composite risk score (from regression model) |
| `obesity_group_normal` | One-hot | Obesity category encoding |
| `obesity_group_overweight` | One-hot | Obesity category encoding |
| `obesity_group_obese` | One-hot | Obesity category encoding |
| `obesity_group_underweight` | One-hot | Obesity category encoding |
| `bp_category_normal` | One-hot | Blood pressure category encoding |
| `bp_category_elevated` | One-hot | Blood pressure category encoding |
| `bp_category_stage_1` | One-hot | Blood pressure category encoding |
| `bp_category_stage_2` | One-hot | Blood pressure category encoding |

---

## 🖥️ Website Pages — Full Walkthrough

### 1. 📊 Dashboard Page

The landing page. Shows how the trained XGBoost model is performing on the test set.

#### KPI Metric Cards (top row)
Five cards summarising model performance on the held-out 20% test set:

| Card | What it means |
|---|---|
| **Accuracy** | Percentage of total predictions (both disease & no-disease) that were correct |
| **Precision** | Of all patients flagged as "disease", how many actually had disease |
| **Recall ★** | Of all patients who truly had disease, how many did the model catch — *the primary target metric, optimised to ≈ 0.87* |
| **F1-Score** | Harmonic mean of Precision and Recall — balances both |
| **ROC-AUC** | Area under the ROC curve; measures model's ability to rank sick vs healthy patients (1.0 = perfect) |

---

#### Confusion Matrix (Plotly Heatmap)
**Model used:** XGBoost (threshold = 0.30)

A 2×2 grid showing:
- **True Negative (top-left):** Healthy patients correctly identified as healthy
- **False Positive (top-right):** Healthy patients incorrectly flagged as diseased
- **False Negative (bottom-left):** Sick patients the model *missed* — we aim to minimise this
- **True Positive (bottom-right):** Sick patients correctly identified

The darker the blue, the higher the count. A good model has large values on the diagonal (top-left and bottom-right).

---

#### ROC Curve (Plotly Line Chart)
**Model used:** XGBoost

Plots the True Positive Rate (Recall) vs the False Positive Rate at every possible decision threshold. The blue line is the model; the dashed grey line is a random baseline. The further the blue curve bends toward the top-left corner, the better. The **AUC score** (shown in the legend) summarises this into a single number.

---

#### Top 15 Feature Importances (Plotly Horizontal Bar Chart)
**Model used:** XGBoost (built-in `.feature_importances_`)

Shows which of the 20 features the XGBoost model relied on most when making predictions. Longer bar = more important. `has_diabetic_retinopathy` is consistently the #1 predictor, as diabetic retinopathy is a direct precursor to eye disease.

---

#### Health Risk Score Distribution (Plotly Histogram)
**Model used:** Linear Regression (used to generate `health_risk_score` for all patients)

Two overlapping histograms — blue for patients without eye disease, red for patients with eye disease. Shows whether higher health risk scores are correlated with higher disease rates, validating the regression model's usefulness as a feature.

---

#### Health Risk Score vs Sugar % (Plotly Scatter Plot)
**Model used:** Linear Regression (health_risk_score axis)

Each dot is one patient, coloured by whether they have eye disease. Shows the relationship between the composite risk score and raw sugar levels, allowing visual inspection of how the regression-derived score separates the two patient groups.

---

### 2. 🔮 Predict Page

The core clinical tool. Enter a patient's metrics and get an instant prediction.

#### Input Form
Collects the following patient data:

**Demographics & Metabolic Markers:**
- Age, Sugar %, Glucose %, Cholesterol %, Obesity %, Heart Rate, Systolic BP, Diastolic BP, Metabolic Risk Count

**Clinical Flags & Categories:**
- Has Diabetic Retinopathy? (Yes/No)
- Metabolic Syndrome Flag? (Yes/No)
- Obesity Group (Normal / Overweight / Obese / Underweight)
- BP Category (Normal / Elevated / Stage 1 / Stage 2)

---

#### Prediction Result Badge
**Model used:** XGBoost Classifier (threshold = 0.30)

After clicking **Run Prediction**:
- Displays either `✅ No Eye Disease Detected` (green) or `⚠️ Eye Disease Detected` (red)
- Shows the **exact probability** percentage (e.g. "67.4% Probability of Eye Disease")
- If probability ≥ 30% → flagged as disease
- If probability < 30% → cleared

---

#### Health Risk Score Gauge
**Model used:** Linear Regression

A Plotly gauge chart (like a speedometer, 0–100) computed in real-time from the patient's entered metabolic values. The needle colour changes with risk level:
- 🟢 Green → Low Risk (0–34)
- 🟡 Yellow → Moderate Risk (35–59)
- 🔴 Red → High Risk (60–100)

The red threshold line at 70 marks the danger zone.

---

#### Clinical Recommendation Box
Rule-based text recommendation generated after prediction:
- **Disease detected** → advises urgent ophthalmologist consultation and DR screening
- **No disease detected** → advises routine annual eye check-ups

---

#### Patient Metabolic Profile Radar Chart
**Model used:** None (visualisation only, based on entered values)

A Plotly polar/radar chart showing six metabolic dimensions for the current patient:
- Sugar %, Glucose %, Cholesterol %, Obesity %, Heart Rate, Health Risk Score

Each axis is normalised to 0–100. A larger filled polygon means higher overall metabolic load. Useful for quickly spotting which factors are elevated for a given patient.

---

### 3. 📈 Analytics Page

Exploratory data analysis (EDA) across the full 20,000-record dataset.

#### Tab 1 — 🔬 Distributions

**Feature Distribution Histogram (Plotly)**
Select any numeric feature from a dropdown. Two overlapping histograms (disease vs no-disease) show how the chosen feature is distributed across both patient groups. Key insight: do the distributions overlap or separate cleanly?

**Box Plot (Plotly)**
For the same selected feature, shows the median, interquartile range, and outliers for each patient class. If the disease group's box sits noticeably higher, it's a strong predictor.

---

#### Tab 2 — 🔗 Correlations

**Feature Correlation Heatmap (Plotly Imshow)**
A 12×12 grid showing Pearson correlation coefficients between the top 12 numeric features. Colour:
- **Deep blue** = strong negative correlation
- **White/grey** = no correlation
- **Deep red** = strong positive correlation

Use this to identify multicollinearity (e.g. systolic and diastolic BP are strongly correlated).

**Interactive Scatter Plot (Plotly)**
Choose any X and Y feature from dropdowns. Each point is a patient, coloured by disease status. Use this to explore pairwise relationships beyond what the heatmap shows.

---

#### Tab 3 — 📊 Group Analysis

**Eye Disease by Obesity Group (Plotly Grouped Bar Chart)**
Side-by-side bars for each obesity category (Normal, Overweight, Obese, Underweight), split by disease status. Shows whether obese patients have disproportionately higher disease rates.

**Eye Disease by Age Group (Plotly Grouped Bar Chart)**
Patients bucketed into age bands: `< 30`, `30–45`, `45–60`, `60+`. Shows the age-risk relationship — older groups typically show higher disease prevalence.

**Health Risk Score Violin Plot (Plotly)**
Violin plots (showing full distribution shape) plus embedded box plots for each disease class. The wider the violin at a particular score, the more patients cluster there. Reveals whether the regression-derived score truly separates the two groups.

---

### 4. ℹ️ About Page

Static informational page covering:
- XGBoost model hyperparameters and rationale for the 0.30 threshold
- Linear Regression health risk score methodology
- Feature engineering explanation
- Medical disclaimer

---

## 🔧 Deployment Notes

### Running locally (no virtual environment needed)
The app runs on the **global Anaconda Python environment**. All required packages are already installed system-wide:

```
streamlit==1.35.0
pandas==2.2.2
numpy==1.26.4
plotly==6.6.0
scikit-learn==1.5.0
xgboost==3.1.3
```

### Command to start
```bash
cd "C:\Users\Admin\OneDrive\Desktop\final"
streamlit run app.py
```

### For a fresh machine (e.g. cloud deployment)
```bash
pip install -r requirements.txt
streamlit run app.py
```

### Environment check
Run this to verify all packages are installed before launching:
```bash
python -c "import streamlit, pandas, numpy, plotly, sklearn, xgboost; print('All OK')"
```

---

## ⚠️ Medical Disclaimer

This application is intended for **research and educational purposes only**. It is **not** a substitute for professional medical diagnosis, advice, or treatment. Always consult a qualified healthcare provider for clinical decisions.

---

## 📝 Development Notes

- The model was developed and hypertuned in `EyeDisease (3).ipynb` using `GridSearchCV` across 32 parameter combinations with 3-fold cross-validation (96 total fits)
- The optimal threshold (0.30) was selected by comparing Recall at multiple thresholds (0.30, 0.32, 0.35) to achieve the target of ≥ 0.87 Recall
- `app.py` re-trains both models at startup using `@st.cache_resource` — models are cached in memory and not retrained on every page interaction
- The `.venv` folder (if present) is an incomplete virtual environment and can be ignored — the app uses the global Python installation
