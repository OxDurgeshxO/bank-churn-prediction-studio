# Bank Churn Prediction Studio

**🏦 AI-Powered Customer Churn Prediction Dashboard**

Live demo: *(coming soon)*

---

## 📌 Project Overview
This machine learning project predicts customer churn in a banking context. The interactive Streamlit dashboard lets you:
- **Predict churn probability** for individual customers in real-time
- **Explore the dataset** through EDA visualizations
- **Compare model performance** across 5 trained classifiers

## 🛠️ Tech Stack
- **Framework**: Streamlit
- **ML**: Scikit-Learn (Random Forest, Logistic Regression), Imbalanced-Learn (SMOTE)
- **Data**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn

## 📊 Models Trained

| Model | Accuracy | Recall | F1-Score | ROC-AUC |
|:---|:---:|:---:|:---:|:---:|
| **Logistic Regression (SMOTE)** | 49.75% | **0.464** | **0.330** | 0.476 |
| Gradient Boosting (SMOTE) | 59.65% | 0.272 | 0.264 | 0.488 |
| **Random Forest (Balanced)** | **62.05%** | 0.240 | 0.252 | **0.493** |
| Baseline Logistic Regression | 73.30% | 0.000 | 0.000 | 0.459 |

> 💡 SMOTE oversampling increased minority-class recall from **0% → 46.4%**

## 🚀 Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 📂 Files
- `app.py` — Streamlit dashboard
- `churn_analysis.py` — Full ML pipeline script
- `Bank_Churn_Classification_Dataset.csv` — 10,000 customer records
- `requirements.txt` — Python dependencies