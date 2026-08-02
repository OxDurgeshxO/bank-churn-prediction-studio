import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
)
from imblearn.over_sampling import SMOTE

warnings.filterwarnings("ignore")

print("=" * 60, flush=True)
print("BANK CUSTOMER CHURN CLASSIFICATION PIPELINE", flush=True)
print("=" * 60, flush=True)

# 1. Load Data
csv_path = "Bank_Churn_Classification_Dataset.csv"
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"Dataset not found at {csv_path}")

df = pd.read_csv(csv_path)
print(f"\n[1/5] Loaded Dataset: {df.shape[0]} rows, {df.shape[1]} columns", flush=True)

# Clean unused columns
cols_to_drop = [c for c in ["Unnamed: 0", "CustomerID"] if c in df.columns]
if cols_to_drop:
    df.drop(columns=cols_to_drop, inplace=True)
    print(f"Dropped non-predictive columns: {cols_to_drop}", flush=True)

# Coerce TotalCharges numeric
if "TotalCharges" in df.columns and df["TotalCharges"].dtype == "object":
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    missing = df["TotalCharges"].isnull().sum()
    if missing > 0:
        print(f"Imputing {missing} missing TotalCharges values with median.", flush=True)
        df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

# Target & feature split
target_col = "Churn"
X = df.drop(columns=[target_col])
y = df[target_col]

churn_counts = y.value_counts(normalize=True) * 100
print(f"\nClass Distribution:\n - Retained (0): {churn_counts.get(0, 0):.2f}%\n - Churned  (1): {churn_counts.get(1, 0):.2f}%", flush=True)

num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
print(f"Numerical Features  ({len(num_cols)}): {num_cols}", flush=True)
print(f"Categorical Features ({len(cat_cols)}): {cat_cols}", flush=True)

# 2. EDA Visualizations
print("\n[2/5] Generating EDA Charts...", flush=True)

# Target Distribution
plt.figure(figsize=(6, 4))
ax = sns.countplot(x=target_col, data=df, palette="Set2")
plt.title("Target Variable Distribution (Churn)", fontsize=12, fontweight="bold")
plt.xlabel("Churn Status (0 = Retained, 1 = Churned)")
plt.ylabel("Count")
for p in ax.patches:
    ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', xytext=(0, 5), textcoords='offset points', fontweight='bold')
plt.tight_layout()
plt.savefig("target_distribution.png", dpi=300, bbox_inches="tight")
plt.close("all")

# Histograms
if num_cols:
    fig, axes = plt.subplots(1, len(num_cols), figsize=(4 * len(num_cols), 3.5))
    if len(num_cols) == 1:
        axes = [axes]
    for i, col in enumerate(num_cols):
        sns.histplot(df[col], kde=True, ax=axes[i], color="teal")
        axes[i].set_title(f"Distribution of {col}", fontsize=10, fontweight="bold")
    plt.tight_layout()
    plt.savefig("histograms.png", dpi=300, bbox_inches="tight")
    plt.close("all")

# Boxplots vs Churn
if num_cols:
    fig, axes = plt.subplots(1, len(num_cols), figsize=(4 * len(num_cols), 3.5))
    if len(num_cols) == 1:
        axes = [axes]
    for i, col in enumerate(num_cols):
        sns.boxplot(x=target_col, y=col, data=df, ax=axes[i], palette="Set2")
        axes[i].set_title(f"{col} by Churn", fontsize=10, fontweight="bold")
    plt.tight_layout()
    plt.savefig("boxplots.png", dpi=300, bbox_inches="tight")
    plt.close("all")

# Correlation Heatmap
plt.figure(figsize=(8, 6))
corr_df = df[num_cols + [target_col]].corr()
sns.heatmap(corr_df, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation Heatmap (Numerical Features)", fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig("correlation_heatmap.png", dpi=300, bbox_inches="tight")
plt.close("all")

print("Saved EDA charts: target_distribution.png, histograms.png, boxplots.png, correlation_heatmap.png", flush=True)

# 3. Train / Test Split & Preprocessing
print("\n[3/5] Preprocessing & Data Splitting...", flush=True)
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

# Feature names post encoding
cat_encoder = preprocessor.named_transformers_["cat"]
cat_feature_names = cat_encoder.get_feature_names_out(cat_cols)
feature_names = num_cols + list(cat_feature_names)

# Apply SMOTE oversampling for training set
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train_proc, y_train)

print(f"Train Set Original Shape: {X_train_proc.shape}, Churn Balance: {dict(pd.Series(y_train).value_counts())}", flush=True)
print(f"Train Set SMOTE Resampled: {X_train_smote.shape}, Churn Balance: {dict(pd.Series(y_train_smote).value_counts())}", flush=True)

# 4. Model Training & Evaluation
print("\n[4/5] Training & Evaluating Classification Models...", flush=True)

models = {
    "Logistic Regression (Baseline Unweighted)": LogisticRegression(random_state=42),
    "Logistic Regression (Balanced Weights)": LogisticRegression(class_weight="balanced", random_state=42),
    "Logistic Regression (SMOTE)": LogisticRegression(random_state=42),
    "Random Forest (Balanced)": RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42),
    "Gradient Boosting (SMOTE)": GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42),
}

results = []
fig_roc, ax_roc = plt.subplots(figsize=(8, 6))

for name, model in models.items():
    if "SMOTE" in name:
        model.fit(X_train_smote, y_train_smote)
    else:
        model.fit(X_train_proc, y_train)

    y_pred = model.predict(X_test_proc)
    y_proba = model.predict_proba(X_test_proc)[:, 1] if hasattr(model, "predict_proba") else y_pred

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_proba)

    results.append({
        "Model": name,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1-Score": f1,
        "ROC-AUC": auc
    })

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    ax_roc.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})")

ax_roc.plot([0, 1], [0, 1], "k--", label="Chance (AUC = 0.50)")
ax_roc.set_xlabel("False Positive Rate")
ax_roc.set_ylabel("True Positive Rate")
ax_roc.set_title("ROC Curves Model Comparison", fontsize=12, fontweight="bold")
ax_roc.legend(loc="lower right", fontsize=8)
plt.tight_layout()
plt.savefig("roc_curves_comparison.png", dpi=300, bbox_inches="tight")
plt.close("all")

results_df = pd.DataFrame(results).sort_values(by="F1-Score", ascending=False)
print("\n--- MODEL PERFORMANCE COMPARISON ---", flush=True)
print(results_df.to_string(index=False), flush=True)

# Save summary report to text file
with open("model_summary.txt", "w", encoding="utf-8") as f:
    f.write("=== MODEL PERFORMANCE COMPARISON ===\n\n")
    f.write(results_df.to_string(index=False))

# 5. Feature Importances (Best Tree Model)
print("\n[5/5] Extracting Feature Importances & Finalizing Visuals...", flush=True)
best_rf = models["Random Forest (Balanced)"]
importances = best_rf.feature_importances_
feat_imp = pd.DataFrame({"Feature": feature_names, "Importance": importances}).sort_values(by="Importance", ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x="Importance", y="Feature", data=feat_imp, palette="viridis")
plt.title("Feature Importance (Random Forest Balanced)", fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig("coefficients.png", dpi=300, bbox_inches="tight")
plt.close("all")

# Confusion Matrix for best F1 model
best_model_name = results_df.iloc[0]["Model"]
best_model = models[best_model_name]
y_pred_best = best_model.predict(X_test_proc)
cm = confusion_matrix(y_test, y_pred_best)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title(f"Confusion Matrix: {best_model_name}", fontsize=11, fontweight="bold")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=300, bbox_inches="tight")
plt.close("all")

print("Saved Model Evaluation Charts: roc_curves_comparison.png, coefficients.png, confusion_matrix.png", flush=True)
print("\nChurn Classification Pipeline completed successfully!", flush=True)