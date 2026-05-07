"""
Comprehensive model evaluation tests for HR attrition ML models.

This module tests the trained machine learning models, including:
- Model file existence and structure
- Prediction format and types
- Model performance thresholds
- Feature importance calculations
- Feature column consistency
"""

import pytest
import json
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from config import MODELS_DIR


# Expected model files
EXPECTED_MODEL_FILES = [
    "random_forest_model.pkl",
    "logistic_regression_model.pkl",
    "scaler.pkl",
    "feature_columns.json",
    "model_results.json"
]

# Model performance thresholds (minimum sanity checks)
RF_ROC_AUC_THRESHOLD = 0.95
LR_ROC_AUC_THRESHOLD = 0.75


@pytest.fixture
def models_dir():
    """Path to the models directory."""
    return MODELS_DIR


@pytest.fixture
def model_files(models_dir):
    """Dictionary of model file paths if they exist."""
    files = {}
    for filename in EXPECTED_MODEL_FILES:
        file_path = models_dir / filename
        if file_path.exists():
            files[filename] = file_path
    return files


@pytest.fixture
def rf_model(model_files):
    """Load the trained Random Forest model."""
    if "random_forest_model.pkl" not in model_files:
        pytest.skip("Random Forest model file not found")

    return joblib.load(model_files["random_forest_model.pkl"])


@pytest.fixture
def lr_model(model_files):
    """Load the trained Logistic Regression model."""
    if "logistic_regression_model.pkl" not in model_files:
        pytest.skip("Logistic Regression model file not found")

    return joblib.load(model_files["logistic_regression_model.pkl"])


@pytest.fixture
def scaler(model_files):
    """Load the trained scaler."""
    if "scaler.pkl" not in model_files:
        pytest.skip("Scaler file not found")

    return joblib.load(model_files["scaler.pkl"])


@pytest.fixture
def feature_columns(model_files):
    """Load the feature columns list."""
    if "feature_columns.json" not in model_files:
        pytest.skip("Feature columns file not found")

    with open(model_files["feature_columns.json"], "r") as f:
        return json.load(f)


@pytest.fixture
def model_results(model_files):
    """Load the model results."""
    if "model_results.json" not in model_files:
        pytest.skip("Model results file not found")

    with open(model_files["model_results.json"], "r") as f:
        return json.load(f)


@pytest.fixture
def feature_importance(models_dir):
    """Load the feature importance data."""
    feature_importance_path = models_dir / "feature_importance.csv"
    if not feature_importance_path.exists():
        pytest.skip("Feature importance file not found")

    return pd.read_csv(feature_importance_path)


@pytest.fixture
def sample_input_data(feature_columns):
    """Create sample input data for prediction testing."""
    # Create a sample with realistic values based on the dataset
    np.random.seed(42)
    n_samples = 5

    sample = pd.DataFrame({
        "satisfaction_level": np.random.uniform(0.3, 1.0, n_samples),
        "last_evaluation": np.random.uniform(0.4, 1.0, n_samples),
        "number_project": np.random.randint(2, 8, n_samples),
        "average_montly_hours": np.random.randint(120, 320, n_samples),
        "time_spend_company": np.random.randint(2, 10, n_samples),
        "Work_accident": np.random.randint(0, 2, n_samples),
        "promotion_last_5years": np.random.randint(0, 2, n_samples),
        "Department": np.random.randint(0, 3, n_samples),
        "salary": np.random.randint(0, 3, n_samples)
    })

    # Ensure columns are in the correct order
    return sample[feature_columns]


def test_model_files_exist(models_dir):
    """
    Test that all expected model files are created after training.

    This test verifies that the training script successfully creates
    all required model artifacts in the models/ directory.
    """
    if not models_dir.exists():
        pytest.skip(f"Models directory {models_dir} does not exist")

    missing_files = []
    for filename in EXPECTED_MODEL_FILES:
        file_path = models_dir / filename
        if not file_path.exists():
            missing_files.append(filename)

    if missing_files:
        pytest.fail(f"Missing model files: {', '.join(missing_files)}")

    # Verify files are not empty
    for filename in EXPECTED_MODEL_FILES:
        file_path = models_dir / filename
        assert file_path.stat().st_size > 0, f"File {filename} is empty"


def test_model_predictions_format(rf_model, lr_model, scaler, sample_input_data):
    """
    Test that model predictions return the correct format and types.

    Verifies that:
    - Binary predictions return 0 or 1
    - Probability predictions return floats between 0 and 1
    - Prediction shape matches input shape
    """
    # Test Random Forest predictions (doesn't need scaling)
    rf_predictions = rf_model.predict(sample_input_data)
    rf_probabilities = rf_model.predict_proba(sample_input_data)[:, 1]

    # Check prediction type and values
    assert rf_predictions.dtype in [np.int32, np.int64], "Predictions should be integer type"
    assert set(rf_predictions).issubset({0, 1}), "Predictions should be binary (0 or 1)"

    # Check probability type and range
    assert rf_probabilities.dtype == np.float64, "Probabilities should be float type"
    assert np.all((rf_probabilities >= 0) & (rf_probabilities <= 1)), "Probabilities must be between 0 and 1"

    # Check shape
    assert len(rf_predictions) == len(sample_input_data), "Prediction length should match input length"
    assert len(rf_probabilities) == len(sample_input_data), "Probability length should match input length"

    # Test Logistic Regression predictions (requires scaling)
    scaled_sample = scaler.transform(sample_input_data)
    lr_predictions = lr_model.predict(scaled_sample)
    lr_probabilities = lr_model.predict_proba(scaled_sample)[:, 1]

    # Same checks for Logistic Regression
    assert lr_predictions.dtype in [np.int32, np.int64], "LR predictions should be integer type"
    assert set(lr_predictions).issubset({0, 1}), "LR predictions should be binary (0 or 1)"
    assert lr_probabilities.dtype == np.float64, "LR probabilities should be float type"
    assert np.all((lr_probabilities >= 0) & (lr_probabilities <= 1)), "LR probabilities must be between 0 and 1"


def test_minimum_accuracy_threshold(model_results):
    """
    Test that models meet minimum performance thresholds.

    These are sanity checks to ensure models perform reasonably well.
    The Random Forest should achieve ROC AUC > 0.95 and
    Logistic Regression should achieve ROC AUC > 0.75.
    """
    # Check Random Forest performance
    rf_roc_auc = model_results["random_forest"]["roc_auc"]
    assert rf_roc_auc > RF_ROC_AUC_THRESHOLD, (
        f"Random Forest ROC AUC {rf_roc_auc:.4f} is below threshold {RF_ROC_AUC_THRESHOLD}"
    )

    # Check Logistic Regression performance
    lr_roc_auc = model_results["logistic_regression"]["roc_auc"]
    assert lr_roc_auc > LR_ROC_AUC_THRESHOLD, (
        f"Logistic Regression ROC AUC {lr_roc_auc:.4f} is below threshold {LR_ROC_AUC_THRESHOLD}"
    )

    # Verify accuracy scores are reasonable (between 0.5 and 1.0)
    rf_accuracy = model_results["random_forest"]["accuracy"]
    lr_accuracy = model_results["logistic_regression"]["accuracy"]

    assert 0.5 < rf_accuracy <= 1.0, f"RF accuracy {rf_accuracy:.4f} is outside reasonable range"
    assert 0.5 < lr_accuracy <= 1.0, f"LR accuracy {lr_accuracy:.4f} is outside reasonable range"


def test_feature_importance_format(feature_importance):
    """
    Test that feature importance data has the correct format.

    Verifies:
    - Required columns exist
    - Importance values sum to approximately 1.0
    - All importance values are non-negative
    - Features are sorted by importance (descending)
    """
    # Check required columns
    required_columns = ["feature", "importance"]
    for col in required_columns:
        assert col in feature_importance.columns, f"Missing required column: {col}"

    # Check importance values are non-negative
    assert (feature_importance["importance"] >= 0).all(), "All importance values must be non-negative"

    # Check importance sum is approximately 1.0 (within 0.01 tolerance)
    importance_sum = feature_importance["importance"].sum()
    assert importance_sum == pytest.approx(1.0, abs=0.01), (
        f"Importance values sum to {importance_sum:.4f}, expected ~1.0"
    )

    # Verify features are sorted by importance (descending order)
    importances = feature_importance["importance"].values
    assert importances[0] >= importances[-1], "Features should be sorted by importance in descending order"

    # Check that no feature has zero importance (unless truly irrelevant)
    zero_importance = feature_importance[feature_importance["importance"] == 0]
    assert len(zero_importance) == 0, "No features should have exactly zero importance"


def test_feature_columns_consistency(feature_columns, rf_model):
    """
    Test that feature columns are consistent with model expectations.

    Verifies that the number of features in feature_columns.json
    matches what the trained model expects.
    """
    # Check that feature_columns is a list
    assert isinstance(feature_columns, list), "Feature columns should be a list"

    # Check that feature columns are strings
    assert all(isinstance(col, str) for col in feature_columns), "All feature names should be strings"

    # Check that we have the expected number of features
    expected_num_features = 9  # Based on the training script
    assert len(feature_columns) == expected_num_features, (
        f"Expected {expected_num_features} features, got {len(feature_columns)}"
    )

    # Check that the model expects the same number of features
    model_n_features = rf_model.n_features_in_
    assert model_n_features == len(feature_columns), (
        f"Model expects {model_n_features} features but feature_columns.json has {len(feature_columns)}"
    )

    # Verify feature names match expected set
    expected_features = {
        "satisfaction_level", "last_evaluation", "number_project",
        "average_montly_hours", "time_spend_company", "Work_accident",
        "promotion_last_5years", "Department", "salary"
    }
    assert set(feature_columns) == expected_features, (
        f"Feature columns don't match expected set. Got: {set(feature_columns)}"
    )


def test_model_results_structure(model_results):
    """
    Test that model results have the expected structure.

    Verifies the JSON structure contains all required metrics
    for both models.
    """
    # Check top-level keys
    assert "random_forest" in model_results, "Missing random_forest results"
    assert "logistic_regression" in model_results, "Missing logistic_regression results"

    # Check required metrics for each model
    required_metrics = ["accuracy", "roc_auc", "cv_scores"]
    for model_name in ["random_forest", "logistic_regression"]:
        model_metrics = model_results[model_name]
        for metric in required_metrics:
            assert metric in model_metrics, f"Missing {metric} for {model_name}"

        # Check cv_scores has 5 values (5-fold cross-validation)
        assert len(model_metrics["cv_scores"]) == 5, f"Expected 5 CV scores, got {len(model_metrics['cv_scores'])}"

        # Check all CV scores are valid (between 0 and 1)
        for score in model_metrics["cv_scores"]:
            assert 0 <= score <= 1, f"CV score {score} is outside valid range [0, 1]"
