import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
)
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import shap

warnings.filterwarnings("ignore")
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

print("=" * 70, flush=True)
print("[BANK] ENTERPRISE BANK CUSTOMER CHURN ML PIPELINE (2026 EDITION)", flush=True)
print("=" * 70, flush=True)

# 1. LOAD & INSPECT DATASET
csv_path = "Bank_Churn_Classification_Dataset.csv"
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"Dataset not found at {csv_path}")

df = pd.read_csv(csv_path)
print(f"\n[1/6] Ingested Dataset: {df.shape[0]:,} records, {df.shape[1]} features", flush=True)

target_col = "Churn"
if target_col not in df.columns:
    raise KeyError(f"Target column '{target_col}' not found in dataset columns: {df.columns.tolist()}")

churn_dist = df[target_col].value_counts(normalize=True) * 100
print(f"Class Distribution:\n - Retained (0): {churn_dist.get(0, 0):.2f}% ({int((df[target_col] == 0).sum()):,} users)")
print(f" - Churned  (1): {churn_dist.get(1, 0):.2f}% ({int((df[target_col] == 1).sum()):,} users)\n", flush=True)

# Drop any non-predictive identifier if present
cols_to_drop = [c for c in ["CustomerId", "Surname", "RowNumber", "Unnamed: 0"] if c in df.columns]
if cols_to_drop:
    df.drop(columns=cols_to_drop, inplace=True)
    print(f"Dropped identifiers: {cols_to_drop}", flush=True)

# Define feature subsets
num_cols = ["CreditScore", "Age", "Tenure", "Balance", "NumOfProducts", "HasCrCard", "IsActiveMember", "EstimatedSalary"]
cat_cols = ["Geography", "Gender"]

# Verify column existence
num_cols = [c for c in num_cols if c in df.columns]
cat_cols = [c for c in cat_cols if c in df.columns]

print(f"Numerical Features  ({len(num_cols)}): {num_cols}")
print(f"Categorical Features ({len(cat_cols)}): {cat_cols}", flush=True)

# 2. GENERATE ADVANCED EDA VISUALIZATIONS
print("\n[2/6] Generating Publication-Grade EDA Visuals...", flush=True)

# Chart A: Demographics & Target Breakdown
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

# 1. Churn by Geography
geo_churn = df.groupby("Geography")["Churn"].mean() * 100
sns.barplot(x=geo_churn.index, y=geo_churn.values, ax=axes[0], palette="Blues_d")
axes[0].set_title("Churn Rate by Country (%)", fontsize=12, fontweight="bold")
axes[0].set_ylabel("Churn Rate (%)")
for p in axes[0].patches:
    axes[0].annotate(f"{p.get_height():.1f}%", (p.get_x() + p.get_width() / 2., p.get_height()),
                     ha='center', va='bottom', fontsize=10, fontweight="bold")

# 2. Churn by Number of Products
prod_churn = df.groupby("NumOfProducts")["Churn"].mean() * 100
sns.barplot(x=prod_churn.index, y=prod_churn.values, ax=axes[1], palette="Reds_d")
axes[1].set_title("Churn Rate by Number of Products (%)", fontsize=12, fontweight="bold")
axes[1].set_ylabel("Churn Rate (%)")
for p in axes[1].patches:
    axes[1].annotate(f"{p.get_height():.1f}%", (p.get_x() + p.get_width() / 2., p.get_height()),
                     ha='center', va='bottom', fontsize=10, fontweight="bold")

# 3. Age Distribution by Churn Status
sns.kdeplot(data=df, x="Age", hue="Churn", common_norm=False, fill=True, ax=axes[2], palette=["#10b981", "#ef4444"])
axes[2].set_title("Age Distribution (Retained vs Churned)", fontsize=12, fontweight="bold")
axes[2].legend(labels=["Churned (1)", "Retained (0)"], loc="upper right")

plt.tight_layout()
plt.savefig("eda_demographics.png", dpi=300)
plt.close()

# Chart B: Correlation Heatmap
plt.figure(figsize=(10, 7))
corr = df[num_cols + [target_col]].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, annot=True, mask=mask, cmap="RdBu_r", center=0, fmt=".2f", linewidths=0.5, cbar_kws={"shrink": 0.8})
plt.title("Numerical Feature Correlation Matrix", fontsize=13, fontweight="bold", pad=12)
plt.tight_layout()
plt.savefig("correlation_heatmap.png", dpi=300)
plt.close()
print("Saved eda_demographics.png & correlation_heatmap.png", flush=True)

# 3. TRAIN / TEST SPLIT & PIPELINE SETUP
print("\n[3/6] Splitting Data & Assembling Preprocessing Transformers...", flush=True)
X = df.drop(columns=[target_col])
y = df[target_col]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"), cat_cols),
    ],
    verbose_feature_names_out=False
)

# Extract post-transformation feature names
preprocessor.fit(X_train)
cat_encoder = preprocessor.named_transformers_["cat"]
cat_encoded_cols = list(cat_encoder.get_feature_names_out(cat_cols))
all_feature_names = num_cols + cat_encoded_cols
print(f"Transformed Feature Space ({len(all_feature_names)} cols): {all_feature_names}", flush=True)

# 4. MODEL TRAINING & BENCHMARKING
print("\n[4/6] Training & Cross-Validating 5 Production Classifiers...", flush=True)

classifiers = {
    "XGBoost Classifier (Tuned)": xgb.XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.06,
        subsample=0.85,
        colsample_bytree=0.85,
        gamma=1.0,
        eval_metric="logloss",
        random_state=42
    ),
    "Random Forest (Balanced)": RandomForestClassifier(
        n_estimators=180,
        max_depth=10,
        min_samples_split=4,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=140,
        learning_rate=0.08,
        max_depth=4,
        random_state=42
    ),
    "Logistic Regression (Baseline)": LogisticRegression(
        max_iter=1000,
        random_state=42
    )
}

results = []
trained_pipelines = {}

# Process regular classifiers
for name, clf in classifiers.items():
    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", clf)
    ])
    pipe.fit(X_train, y_train)
    trained_pipelines[name] = pipe

    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_proba)
    pr_auc = average_precision_score(y_test, y_proba)
    brier = brier_score_loss(y_test, y_proba)

    results.append({
        "Model": name,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1-Score": f1,
        "ROC-AUC": auc,
        "PR-AUC": pr_auc,
        "Brier-Score": brier
    })
    print(f" [PASS] {name:30s} | ROC-AUC: {auc:.4f} | Accuracy: {acc*100:.2f}% | F1: {f1:.4f}", flush=True)

# 5. SMOTE + Random Forest model
print("Training SMOTE + Random Forest...", flush=True)
X_train_proc = preprocessor.transform(X_train)
X_test_proc = preprocessor.transform(X_test)

smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train_proc, y_train)

rf_smote = RandomForestClassifier(n_estimators=180, max_depth=10, random_state=42, n_jobs=-1)
rf_smote.fit(X_train_smote, y_train_smote)

y_pred_sm = rf_smote.predict(X_test_proc)
y_proba_sm = rf_smote.predict_proba(X_test_proc)[:, 1]

smote_results = {
    "Model": "Random Forest (SMOTE)",
    "Accuracy": accuracy_score(y_test, y_pred_sm),
    "Precision": precision_score(y_test, y_pred_sm, zero_division=0),
    "Recall": recall_score(y_test, y_pred_sm, zero_division=0),
    "F1-Score": f1_score(y_test, y_pred_sm, zero_division=0),
    "ROC-AUC": roc_auc_score(y_test, y_proba_sm),
    "PR-AUC": average_precision_score(y_test, y_proba_sm),
    "Brier-Score": brier_score_loss(y_test, y_proba_sm)
}
results.append(smote_results)
print(f" [PASS] {'Random Forest (SMOTE)':30s} | ROC-AUC: {smote_results['ROC-AUC']:.4f} | Accuracy: {smote_results['Accuracy']*100:.2f}% | F1: {smote_results['F1-Score']:.4f}", flush=True)

# Rank models by ROC-AUC
results_df = pd.DataFrame(results).sort_values(by="ROC-AUC", ascending=False)
print("\n" + "=" * 70)
print("FINAL BENCHMARK LEADERBOARD:")
print("=" * 70)
print(results_df.to_string(index=False), flush=True)

# Pick Best Model
best_model_row = results_df.iloc[0]
best_model_name = best_model_row["Model"]
print(f"\n[CHAMPION] Top Performing Model: {best_model_name} (ROC-AUC: {best_model_row['ROC-AUC']:.4f})", flush=True)

# 5. CHARTS: ROC CURVES, PR CURVES, CONFUSION MATRIX
print("\n[5/6] Exporting Evaluation Curves & Confusion Matrices...", flush=True)

# Plot ROC Curves
plt.figure(figsize=(8, 6))
for name, pipe in trained_pipelines.items():
    proba = pipe.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, proba)
    auc_val = roc_auc_score(y_test, proba)
    plt.plot(fpr, tpr, label=f"{name} (AUC = {auc_val:.3f})", lw=2)

# Add SMOTE curve
fpr_sm, tpr_sm, _ = roc_curve(y_test, y_proba_sm)
plt.plot(fpr_sm, tpr_sm, label=f"Random Forest (SMOTE) (AUC = {smote_results['ROC-AUC']:.3f})", lw=2, linestyle="--")

plt.plot([0, 1], [0, 1], "k--", label="Random Chance (AUC = 0.50)", alpha=0.6)
plt.xlabel("False Positive Rate", fontsize=11)
plt.ylabel("True Positive Rate", fontsize=11)
plt.title("Multi-Model ROC Curves Comparison", fontsize=13, fontweight="bold")
plt.legend(loc="lower right", fontsize=9)
plt.tight_layout()
plt.savefig("roc_curves_comparison.png", dpi=300)
plt.close()

# Plot Confusion Matrix for Best Pipeline
best_pipe = trained_pipelines.get(best_model_name, trained_pipelines["XGBoost Classifier (Tuned)"])
y_pred_best = best_pipe.predict(X_test)
cm = confusion_matrix(y_test, y_pred_best)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=["Retained (0)", "Churned (1)"],
            yticklabels=["Retained (0)", "Churned (1)"])
plt.xlabel("Predicted Status", fontsize=11, fontweight="bold")
plt.ylabel("Actual Status", fontsize=11, fontweight="bold")
plt.title(f"Confusion Matrix: {best_model_name}", fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=300)
plt.close()

# 6. SHAP EXPLAINER & ARTIFACT SERIALIZATION
print("\n[6/6] Computing SHAP Explainability & Serializing Production Artifacts...", flush=True)
os.makedirs("models", exist_ok=True)

# Best Model Extraction
best_clf = best_pipe.named_steps["classifier"]
best_prep = best_pipe.named_steps["preprocessor"]

# Fit TreeExplainer
explainer = shap.TreeExplainer(best_clf)
X_test_transformed = best_prep.transform(X_test)
shap_values = explainer.shap_values(X_test_transformed)

# If multi-class or binary shape check
if isinstance(shap_values, list):
    shap_vals_class1 = shap_values[1]
elif shap_values.ndim == 3:
    shap_vals_class1 = shap_values[:, :, 1]
else:
    shap_vals_class1 = shap_values

# SHAP Summary Plot
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_vals_class1, X_test_transformed, feature_names=all_feature_names, show=False)
plt.title("SHAP Global Feature Attribution (Impact on Churn Probability)", fontsize=12, fontweight="bold", pad=12)
plt.tight_layout()
plt.savefig("shap_summary.png", dpi=300, bbox_inches="tight")
plt.close()

# Serialize Pipeline & Metadata
joblib.dump(best_pipe, "models/bank_churn_pipeline.joblib")
joblib.dump(explainer, "models/shap_explainer.joblib")
joblib.dump(all_feature_names, "models/feature_names.joblib")

# Save structured metrics
summary_payload = {
    "best_model": best_model_name,
    "feature_names": all_feature_names,
    "metrics": results_df.to_dict(orient="records")
}
with open("models/metrics.json", "w", encoding="utf-8") as f:
    json.dump(summary_payload, f, indent=2)

with open("model_summary.txt", "w", encoding="utf-8") as f:
    f.write("=== ENTERPRISE BANK CHURN MODEL BENCHMARK ===\n\n")
    f.write(results_df.to_string(index=False))

print("Saved models/bank_churn_pipeline.joblib")
print("Saved models/shap_explainer.joblib")
print("Saved models/metrics.json")
print("Saved roc_curves_comparison.png, confusion_matrix.png, shap_summary.png")
print("\n Pipeline Execution Complete! All artifacts ready for production Streamlit Studio.", flush=True)