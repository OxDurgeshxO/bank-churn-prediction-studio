# 🏦 Bank Churn Prediction Studio

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://bank-churn-prediction-studio-mrl8whyxpnhkyfvfwmtqwq.streamlit.app/)
![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/Model-XGBoost%20Tuned-FF6600?logo=xgboost&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/ML-Scikit--Learn-F7931E?logo=scikit-learn&logoColor=white)
![SHAP](https://img.shields.io/badge/Explainability-SHAP%20XAI-green)
![ROC-AUC](https://img.shields.io/badge/ROC--AUC-0.871-success)
![Accuracy](https://img.shields.io/badge/Accuracy-86.65%25-brightgreen)

> **An Enterprise FinTech Decision Support System combining gradient-boosted attrition forecasting, real-time SHAP explainability, and an executive retention campaign ROI simulator.**

---

## 📌 Executive Overview

Customer attrition in retail banking represents millions of dollars in lost Customer Lifetime Value (CLV). **Bank Churn Prediction Studio** translates complex predictive modeling into clear business decisions:
- **Sub-10ms Inference**: Evaluates customer flight risk instantly using pre-trained and serialized production pipelines.
- **Explainable AI (SHAP Waterfall)**: Breaks down every prediction into positive and negative risk contributions (e.g., age cohort, inactivity status, regional market dynamics).
- **Prescriptive Retention Playbook**: Generates targeted, personalized interventions (fee waivers, concierge check-ins, interest rate matching).
- **Executive ROI Simulator**: Models customer cohort size, retention campaign costs, and preserved CLV to compute net profit and campaign ROI before capital deployment.

---

## 🏗️ Architecture & Pipeline Flow

```
   10,000 Verified Bank Records
                 │
                 ▼
┌─────────────────────────────────┐
│     Data Pipeline & Scaling     │
│  StandardScaler + OneHotEncoder │
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│     Multi-Model Benchmark       │
│  XGBoost • Random Forest • SMOTE│
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│  Serialization & Explainability │
│  .joblib Pipeline + SHAP Engine │
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────┐
│              Streamlit Enterprise Cockpit               │
│  [1] Real-Time XAI Diagnostic                           │
│  [2] Executive Retention Campaign ROI Simulator         │
│  [3] Behavioral Exploratory Data Analytics (EDA)        │
│  [4] Multi-Classifier Benchmark & Confusion Telemetry   │
└─────────────────────────────────────────────────────────┘
```

---

## 📊 Rigorous Model Benchmarks

Trained and evaluated on 10,000 customer banking records with stratified test splits:

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | PR-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 🏆 **XGBoost Classifier (Tuned)** | **86.65%** | **0.784** | 0.485 | **0.600** | **0.871** | **0.698** |
| 🥈 **Gradient Boosting** | 86.35% | 0.768 | 0.480 | 0.591 | **0.866** | 0.691 |
| 🥉 **Random Forest (Balanced)** | 85.10% | 0.665 | **0.575** | **0.617** | **0.865** | 0.684 |
| 📈 **Random Forest (SMOTE)** | 84.75% | 0.648 | **0.582** | 0.613 | 0.858 | 0.672 |
| 📉 **Logistic Regression (Baseline)** | 81.10% | 0.590 | 0.198 | 0.297 | 0.772 | 0.482 |

---

## 🔍 Key Behavioral Insights (EDA)

1. **Age Inflection**: Churn probability sharply escalates between ages **45–60**, indicating wealth transfer and retirement-driven account consolidations.
2. **Product Paradox**: Customers with **2 products** demonstrate the highest retention (only ~8% churn), whereas holding **3 or 4 products** spikes churn past **80%** due to fee dissatisfaction.
3. **Regional Sensitivity**: German branch customers exhibit roughly double the baseline churn rate (~32%) compared to France and Spain (~16%), requiring localized interest rate matching.

---

## 🚀 Quickstart & Local Setup

### 1. Clone & Setup Virtual Environment
```bash
git clone https://github.com/OxDurgeshxO/bank-churn-prediction-studio.git
cd bank-churn-prediction-studio

python -m venv .venv
# On Windows
.\.venv\Scripts\activate
# On macOS/Linux
source .venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Pipeline & Launch Studio
```bash
# Execute ML pipeline, serialize artifacts, and generate charts
python churn_analysis.py

# Launch interactive Streamlit studio
streamlit run app.py
```

---

## 📂 Repository Structure

```
├── Bank_Churn_Classification_Dataset.csv # 10,000 customer banking records
├── app.py                                # Streamlit Enterprise Cockpit (4 Tabs)
├── churn_analysis.py                     # Offline ML pipeline & SHAP serialization
├── models/
│   ├── bank_churn_pipeline.joblib        # Serialized preprocessor + champion XGBoost
│   ├── shap_explainer.joblib             # Serialized SHAP TreeExplainer
│   ├── feature_names.joblib              # Transformed column metadata
│   └── metrics.json                      # Benchmark leaderboard data
├── roc_curves_comparison.png             # Multi-model ROC comparison plot
├── confusion_matrix.png                  # Champion model confusion matrix
├── shap_summary.png                      # Global SHAP beeswarm feature attribution
├── eda_demographics.png                  # Behavioral churn breakdown
├── correlation_heatmap.png               # Inter-feature correlation matrix
├── requirements.txt                      # Project dependencies
└── readme.md                             # Documentation
```

---

## 👤 Author
**Durgesh**  
GitHub: [@OxDurgeshxO](https://github.com/OxDurgeshxO)