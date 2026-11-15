import os
import pytest
import pandas as pd
import joblib

def test_model_artifacts_exist():
    assert os.path.exists("models/bank_churn_pipeline.joblib"), "Trained pipeline missing"
    assert os.path.exists("models/feature_names.joblib"), "Feature names missing"

def test_dataset_exists():
    assert os.path.exists("Bank_Churn_Classification_Dataset.csv"), "Dataset missing"
    df = pd.read_csv("Bank_Churn_Classification_Dataset.csv")
    assert not df.empty, "Dataset must not be empty"

def test_pipeline_loading():
    pipeline = joblib.load("models/bank_churn_pipeline.joblib")
    assert pipeline is not None, "Pipeline must load properly"
