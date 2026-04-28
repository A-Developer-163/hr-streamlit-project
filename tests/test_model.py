"""
Test ML model functionality
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path


@pytest.fixture
def model_artifacts():
    """Load model artifacts if they exist."""
    import joblib
    import json

    models_dir = Path("models")
    if not models_dir.exists():
        pytest.skip("Model artifacts not found. Run train_model.py first")

    artifacts = {}
    try:
        artifacts["rf_model"] = joblib.load(models_dir / "random_forest_model.pkl")
        artifacts["lr_model"] = joblib.load(models_dir / "logistic_regression_model.pkl")
        artifacts["scaler"] = joblib.load(models_dir / "scaler.pkl")
        artifacts["le_dept"] = joblib.load(models_dir / "label_encoder_department.pkl")
        artifacts["le_salary"] = joblib.load(models_dir / "label_encoder_salary.pkl")

        with open(models_dir / "feature_columns.json", "r") as f:
            artifacts["feature_cols"] = json.load(f)

        with open(models_dir / "model_results.json", "r") as f:
            artifacts["results"] = json.load(f)

        return artifacts
    except Exception as e:
        pytest.skip(f"Could not load model artifacts: {e}")


def test_model_artifacts_exist(model_artifacts):
    """Test that all model artifacts are loaded."""
    assert "rf_model" in model_artifacts
    assert "lr_model" in model_artifacts
    assert "scaler" in model_artifacts
    assert "feature_cols" in model_artifacts


def test_random_forest_prediction(model_artifacts):
    """Test Random Forest model makes predictions."""
    model = model_artifacts["rf_model"]
    feature_cols = model_artifacts["feature_cols"]

    # Create sample input
    sample = np.array([[0.5, 0.7, 4, 200, 3, 0, 0, 1, 1]])  # All features

    prediction = model.predict(sample)
    probability = model.predict_proba(sample)

    assert prediction.shape == (1,)
    assert probability.shape == (1, 2)
    assert probability[0].sum() == pytest.approx(1.0, 0.01)


def test_logistic_regression_prediction(model_artifacts):
    """Test Logistic Regression model makes predictions."""
    model = model_artifacts["lr_model"]
    scaler = model_artifacts["scaler"]

    # Create sample input
    sample = np.array([[0.5, 0.7, 4, 200, 3, 0, 0, 1, 1]])

    # Scale features
    sample_scaled = scaler.transform(sample)

    prediction = model.predict(sample_scaled)
    probability = model.predict_proba(sample_scaled)

    assert prediction.shape == (1,)
    assert probability.shape == (1, 2)
    assert probability[0].sum() == pytest.approx(1.0, 0.01)


def test_feature_importance(model_artifacts):
    """Test feature importance is available."""
    models_dir = Path("models")
    feature_importance = pd.read_csv(models_dir / "feature_importance.csv")

    assert "feature" in feature_importance.columns
    assert "importance" in feature_importance.columns
    assert len(feature_importance) > 0
    assert feature_importance["importance"].sum() == pytest.approx(1.0, 0.1)


def test_model_performance_thresholds(model_artifacts):
    """Test models meet minimum performance thresholds."""
    results = model_artifacts["results"]

    # Random Forest should have high accuracy
    rf_accuracy = results["random_forest"]["accuracy"]
    assert rf_accuracy > 0.95, f"Random Forest accuracy {rf_accuracy} below threshold 0.95"

    # ROC AUC should be good
    rf_auc = results["random_forest"]["roc_auc"]
    assert rf_auc > 0.95, f"Random Forest AUC {rf_auc} below threshold 0.95"


def test_label_encoders(model_artifacts):
    """Test label encoders work correctly."""
    le_dept = model_artifacts["le_dept"]
    le_salary = model_artifacts["le_salary"]

    # Test encoding and decoding
    dept = "sales"
    encoded = le_dept.transform([dept])[0]
    decoded = le_dept.inverse_transform([encoded])[0]

    assert decoded == dept

    salary = "medium"
    encoded_salary = le_salary.transform([salary])[0]
    decoded_salary = le_salary.inverse_transform([encoded_salary])[0]

    assert decoded_salary == salary
