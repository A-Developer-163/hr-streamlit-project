#!/usr/bin/env python3
"""
Generate SHAP Explainer for Model Explainability
Creates and saves SHAP explainer and values for the trained attrition model.
Run this after train_model.py to generate explainability artifacts.
"""

import pandas as pd
import numpy as np
import joblib
import json
import shap
import time
from pathlib import Path
from config import MODELS_DIR, HR_DATA_PATH


def load_artifacts(models_path: Path = None):
    """Load trained model and preprocessing artifacts.

    Args:
        models_path: Path to models directory (defaults to config.MODELS_DIR)
    """
    if models_path is None:
        models_path = MODELS_DIR
    print("Loading model artifacts...")

    rf_model = joblib.load(models_path / "rf_attrition_model.pkl")
    le_department = joblib.load(models_path / "department_encoder.pkl")
    le_salary = joblib.load(models_path / "salary_encoder.pkl")

    with open(models_path / "feature_columns.json", "r") as f:
        feature_cols = json.load(f)

    with open(models_path / "model_results.json", "r") as f:
        model_results = json.load(f)

    print(f"  [OK] Random Forest model loaded")
    print(f"  [OK] Label encoders loaded")
    print(f"  [OK] Feature columns loaded: {len(feature_cols)} features")

    return rf_model, le_department, le_salary, feature_cols, model_results


def load_data(data_path: Path = None):
    """Load and preprocess HR employee data.

    Args:
        data_path: Path to data file (defaults to config.HR_DATA_PATH)
    """
    if data_path is None:
        data_path = Path(HR_DATA_PATH)
    print("\nLoading data...")
    df = pd.read_csv(data_path)
    df = df.drop("Emp_Id", axis=1)

    print(f"  [OK] Data loaded: {len(df)} employees")
    print(f"  Attrition rate: {df['left'].mean() * 100:.1f}%")

    return df


def prepare_features(df, le_department, le_salary, feature_cols):
    """Encode categorical features and prepare feature matrix."""
    print("\nPreparing features...")

    X = df.copy()

    # Encode salary (ordinal) - use 'salary' column name to match training
    X["salary"] = le_salary.transform(X[["salary"]]).flatten()

    # Encode department (one-hot)
    dept_encoded = le_department.transform(X[["Department"]])
    dept_columns = [f'Dept_{cat}' for cat in le_department.categories_[0][1:]]
    for i, col in enumerate(dept_columns):
        X[col] = dept_encoded[:, i]

    # Select only the feature columns
    X = X[feature_cols].copy()
    y = df["left"]

    print(f"  [OK] Dataset shape: {X.shape}")

    return X, y


def create_explainer(rf_model, X_sample, sample_size: int = 1000):
    """Create SHAP TreeExplainer and compute SHAP values."""
    print("\nInitializing SHAP TreeExplainer...")
    explainer = shap.TreeExplainer(rf_model)
    print("  [OK] TreeExplainer initialized")

    # Sample data for SHAP computation
    if len(X_sample) > sample_size:
        print(f"\nComputing SHAP values for {sample_size} random samples...")
        sample_idx = np.random.choice(len(X_sample), sample_size, replace=False)
        X_shap = X_sample.iloc[sample_idx].copy()
    else:
        print(f"\nComputing SHAP values for all {len(X_sample)} samples...")
        X_shap = X_sample.copy()

    print("  This may take 30-60 seconds. Please wait...")
    start_time = time.time()

    shap_values = explainer.shap_values(X_shap)

    elapsed = time.time() - start_time
    print(f"  [OK] SHAP computation completed in {elapsed:.1f} seconds")

    # Extract SHAP values for attrition class (class 1)
    if isinstance(shap_values, list):
        shap_values_attrition = shap_values[1]
        print(f"  [OK] SHAP values (list format): {shap_values_attrition.shape}")
    elif len(shap_values.shape) == 3:
        shap_values_attrition = shap_values[:, :, 1]
        print(f"  [OK] SHAP values (3D array format): {shap_values_attrition.shape}")
    else:
        shap_values_attrition = shap_values
        print(f"  [OK] SHAP values (2D array format): {shap_values_attrition.shape}")

    # Extract base value
    if isinstance(explainer.expected_value, list):
        base_val = explainer.expected_value[1]
    elif len(explainer.expected_value) > 1:
        base_val = explainer.expected_value[1]
    else:
        base_val = explainer.expected_value

    print(f"  [OK] Base value (average attrition risk): {base_val:.4f}")

    return explainer, shap_values_attrition, X_shap, base_val


def calculate_feature_importance(shap_values, feature_cols):
    """Calculate mean absolute SHAP values for feature importance."""
    mean_shap = pd.DataFrame({
        'feature': feature_cols,
        'mean_abs_shap': np.abs(shap_values).mean(axis=0)
    }).sort_values('mean_abs_shap', ascending=False)

    return mean_shap


def save_explainer_artifacts(explainer, shap_values, X_shap, mean_shap,
                              base_val, models_path: Path = Path("models")):
    """Save SHAP explainer and related artifacts."""
    import pickle

    print("\nSaving SHAP artifacts...")

    # Save explainer
    with open(models_path / 'shap_explainer.pkl', 'wb') as f:
        pickle.dump(explainer, f)
    print(f"  [OK] SHAP explainer saved to {models_path / 'shap_explainer.pkl'}")

    # Save SHAP values
    np.save(models_path / 'shap_values_sample.npy', shap_values)
    print(f"  [OK] SHAP values saved to {models_path / 'shap_values_sample.npy'}")

    # Save sample indices
    np.save(models_path / 'shap_sample_indices.npy', X_shap.index.values)
    print(f"  [OK] Sample indices saved to {models_path / 'shap_sample_indices.npy'}")

    # Save feature importance
    mean_shap.to_csv(models_path / 'shap_feature_importance.csv', index=False)
    print(f"  [OK] Feature importance saved to {models_path / 'shap_feature_importance.csv'}")

    # Save base value
    with open(models_path / 'shap_base_value.json', 'w') as f:
        json.dump({'base_value': base_val}, f)
    print(f"  [OK] Base value saved to {models_path / 'shap_base_value.json'}")


def print_summary(mean_shap, model_results):
    """Print summary of SHAP analysis."""
    print("\n" + "=" * 70)
    print("SHAP FEATURE IMPORTANCE RANKING")
    print("=" * 70)
    print(mean_shap.to_string(index=False))

    print("\n" + "=" * 70)
    print("ARTIFACTS SAVED")
    print("=" * 70)
    print("""
The following files are now available in models/:
  - shap_explainer.pkl           - SHAP explainer object
  - shap_values_sample.npy       - SHAP values for sample
  - shap_sample_indices.npy      - Indices of sampled employees
  - shap_feature_importance.csv  - Mean |SHAP| by feature
  - shap_base_value.json         - Base value for predictions

Use these artifacts in:
  - notebooks/03_model_explainability.ipynb
  - scripts/explain_with_shap.py
  - Streamlit app for live predictions
    """)


def main():
    print("=" * 70)
    print("SHAP EXPLAINER GENERATION")
    print("=" * 70)

    models_path = MODELS_DIR
    data_path = Path(HR_DATA_PATH)

    # Load artifacts
    rf_model, le_dept, le_sal, feature_cols, model_results = load_artifacts(models_path)

    # Load data
    df = load_data(data_path)

    # Prepare features
    X, y = prepare_features(df, le_dept, le_sal, feature_cols)

    # Create explainer and compute SHAP values
    explainer, shap_values, X_shap, base_val = create_explainer(rf_model, X)

    # Calculate feature importance
    mean_shap = calculate_feature_importance(shap_values, feature_cols)

    # Save artifacts
    save_explainer_artifacts(explainer, shap_values, X_shap, mean_shap, base_val, models_path)

    # Print summary
    print_summary(mean_shap, model_results)


if __name__ == "__main__":
    main()
