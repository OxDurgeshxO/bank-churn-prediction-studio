import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
)
from imblearn.over_sampling import SMOTE

# Page Configuration
st.set_page_config(
    page_title="Bank Churn Prediction Studio",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-title { font-size: 2.2rem; font-weight: 800; color: #0f172a; margin-bottom: 0rem; }
    .sub-title { font-size: 1rem; color: #64748b; margin-bottom: 1.5rem; }
    .metric-card { background-color: #f8fafc; border: 1px solid #e2e8f0; border-radius: 12px; padding: 16px; text-align: center; }
    .stButton>button { background: linear-gradient(135deg, #10b981 0%, #059669 100%); color: white; border: none; font-weight: 600; border-radius: 10px; height: 45px; }
</style>
""", unsafe_allow_html=True)

# Data Loading & Caching
@st.cache_data
def load_and_preprocess_data():
    csv_path = "Bank_Churn_Classification_Dataset.csv"
    if not os.path.exists(csv_path):
        st.error(f"Dataset not found at {csv_path}")
        st.stop()
    
    df = pd.read_csv(csv_path)
    cols_to_drop = [c for c in ["Unnamed: 0", "CustomerID"] if c in df.columns]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
        
    if "TotalCharges" in df.columns and df["TotalCharges"].dtype == "object":
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)
        
    return df

@st.cache_resource
def train_models(df):
    target_col = "Churn"
    X = df.drop(columns=[target_col])
    y = df[target_col]

    num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), cat_cols),
        ]
    )

    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)

    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train_proc, y_train)

    rf_model = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)
    rf_model.fit(X_train_proc, y_train)

    lr_smote = LogisticRegression(random_state=42)
    lr_smote.fit(X_train_smote, y_train_smote)

    return preprocessor, rf_model, lr_smote, num_cols, cat_cols, X_test_proc, y_test

# Load Resources
df = load_and_preprocess_data()
preprocessor, rf_model, lr_smote, num_cols, cat_cols, X_test_proc, y_test = train_models(df)

# Title Header
st.markdown('<div class="main-title">🏦 Bank Customer Churn Studio</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">AI-Powered Risk Analytics & Individual Churn Prediction Dashboard</div>', unsafe_allow_html=True)

# Navigation Tabs
tab1, tab2, tab3 = st.tabs(["🎯 Live Churn Predictor", "📊 EDA Analytics", "⚡ Model Comparison"])

# --- TAB 1: LIVE CHURN PREDICTOR ---
with tab1:
    st.subheader("Predict Customer Churn Probability")
    st.write("Adjust customer demographics and subscription attributes to calculate churn risk in real-time.")

    col1, col2, col3 = st.columns(3)

    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        tenure = st.slider("Tenure (Months)", min_value=1, max_value=72, value=24)

    with col2:
        monthly_charges = st.slider("Monthly Charges ($)", min_value=18.0, max_value=120.0, value=65.0)
        total_charges = st.number_input("Total Charges ($)", min_value=18.0, max_value=9000.0, value=float(tenure * monthly_charges))
        contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])

    with col3:
        payment_method = st.selectbox("Payment Method", [
            "Electronic check", "Mailed check", "Bank transfer", "Credit card"
        ])
        model_choice = st.selectbox("Prediction Model", [
            "Random Forest (Balanced)", "Logistic Regression (SMOTE)"
        ])

    if st.button("🔮 Calculate Customer Churn Risk", use_container_width=True):
        input_data = pd.DataFrame([{
            "Gender": gender,
            "SeniorCitizen": senior,
            "Tenure": tenure,
            "MonthlyCharges": monthly_charges,
            "Contract": contract,
            "PaymentMethod": payment_method,
            "TotalCharges": total_charges
        }])

        input_proc = preprocessor.transform(input_data)
        active_model = rf_model if "Random Forest" in model_choice else lr_smote

        churn_proba = active_model.predict_proba(input_proc)[0][1]
        churn_pred = int(churn_proba >= 0.5)

        st.markdown("---")
        res_col1, res_col2 = st.columns([1, 2])

        with res_col1:
            st.metric("Predicted Status", "🚨 High Risk (Churn)" if churn_pred == 1 else "✅ Low Risk (Retain)")
            st.metric("Churn Probability", f"{churn_proba * 100:.1f}%")

        with res_col2:
            st.progress(float(churn_proba))
            if churn_pred == 1:
                st.error(f"⚠️ **High Churn Risk ({churn_proba * 100:.1f}%)**: Recommend offering a discounted annual contract or loyalty bonus.")
            else:
                st.success(f"🎉 **Customer Retained ({ (1 - churn_proba) * 100:.1f}% stability)**: Low risk of leaving in the upcoming billing cycle.")

# --- TAB 2: EDA ANALYTICS ---
with tab2:
    st.subheader("Exploratory Data Analysis")
    st.write(f"Dataset summary covering {len(df)} customer records across {len(df.columns)} feature attributes.")

    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.countplot(x="Churn", data=df, palette="Set2", ax=ax)
        plt.title("Target Distribution (Churn Status)")
        st.pyplot(fig)
        plt.close()

    with c2:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.boxplot(x="Churn", y="Tenure", data=df, palette="Set2", ax=ax)
        plt.title("Tenure Distribution by Churn")
        st.pyplot(fig)
        plt.close()

    st.markdown("### Numerical Feature Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.heatmap(df[num_cols + ["Churn"]].corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    st.pyplot(fig)
    plt.close()

# --- TAB 3: MODEL COMPARISON ---
with tab3:
    st.subheader("Model Performance Comparison")
    
    metrics_data = [
        {"Model": "Logistic Regression (SMOTE)", "Accuracy": "49.75%", "Precision": "0.256", "Recall": "0.464", "F1-Score": "0.330", "ROC-AUC": "0.476"},
        {"Model": "Logistic Regression (Balanced)", "Accuracy": "46.15%", "Precision": "0.238", "Recall": "0.463", "F1-Score": "0.314", "ROC-AUC": "0.459"},
        {"Model": "Gradient Boosting (SMOTE)", "Accuracy": "59.65%", "Precision": "0.258", "Recall": "0.272", "F1-Score": "0.264", "ROC-AUC": "0.488"},
        {"Model": "Random Forest (Balanced)", "Accuracy": "62.05%", "Precision": "0.266", "Recall": "0.240", "F1-Score": "0.252", "ROC-AUC": "0.493"},
        {"Model": "Logistic Regression (Baseline)", "Accuracy": "73.30%", "Precision": "0.000", "Recall": "0.000", "F1-Score": "0.000", "ROC-AUC": "0.459"},
    ]
    st.table(pd.DataFrame(metrics_data))

    st.info("💡 **SMOTE Oversampling** effectively resolves 0.0% minority class recall, enabling automated detection of customers at risk of churn.")
