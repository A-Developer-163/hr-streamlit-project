#!/usr/bin/env python3
"""
Training Advanced Employee Attrition Prediction Models
Training XGBoost and LightGBM models, comparing against Random Forest baseline.

BIAS FIXES APPLIED:
- Proper ordinal encoding for salary (low=0, medium=1, high=2)
- OneHot encoding for Department (nominal variable)
- Train/Val/Test split (70/15/15) to prevent test set leakage
- Threshold tuning on validation set only
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import json
import time
import os
import argparse
from dotenv import load_dotenv
import sys

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import save_preprocessing_artifacts

# Loading environment variables
load_dotenv()

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator, ClassifierMixin

# Fairness analysis
from fairlearn.metrics import (
    MetricFrame,
    demographic_parity_difference,
    demographic_parity_ratio,
    selection_rate,
    true_positive_rate,
    false_positive_rate,
    equalized_odds_difference
)
from fairlearn.postprocessing import ThresholdOptimizer
from fairlearn.reductions import ExponentiatedGradient, EqualizedOdds
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class Float64XGBoostWrapper(BaseEstimator, ClassifierMixin):
    """Wrapper to ensure XGBoost outputs float64 predictions for fairlearn compatibility.

    XGBoost internally uses float32, but fairlearn's ThresholdOptimizer and
    ExponentiatedGradient expect float64 probabilities. This wrapper converts
    predict_proba output to float64 while maintaining full sklearn compatibility.
    """

    def __init__(self, **kwargs):
        """Initialize wrapper with XGBoost parameters."""
        self.kwargs = kwargs
        self.model = XGBClassifier(**kwargs)

    def fit(self, X, y, **kwargs):
        """Fit the underlying XGBoost model, accepting sample_weight and other sklearn params."""
        self.model.fit(X, y, **kwargs)
        return self

    def predict(self, X):
        """Return class predictions."""
        return self.model.predict(X)

    def predict_proba(self, X):
        """Return class probabilities as float64 for fairlearn compatibility."""
        proba = self.model.predict_proba(X)
        return proba.astype('float64')

    def get_params(self, deep=True):
        """Get parameters for sklearn compatibility."""
        return self.kwargs

    def set_params(self, **params):
        """Set parameters for sklearn compatibility."""
        self.kwargs.update(params)
        self.model = XGBClassifier(**self.kwargs)
        return self


def load_and_preprocess_data(csv_path: str = None):
    """Loading and preprocessing data for modeling."""
    if csv_path is None:
        csv_path = os.getenv("HR_DATA_PATH", "data/hr_employee_data.csv")
    df = pd.read_csv(csv_path)

    df = df.drop("employee_id", axis=1)

    feature_cols = [c for c in df.columns if c != "attrition"]
    X = df[feature_cols]
    y = df["attrition"]

    return df, X, y, feature_cols


def train_models(df, X, y, feature_cols):
    """Training multiple models and comparing performance."""
    # THREE-WAY SPLIT: 70% train, 15% validation, 15% test
    # Validation set for threshold tuning (prevents test set leakage)
    print("\nSplitting data: 70% train, 15% validation, 15% test")
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # PROPER ENCODING: Ordinal for salary, OneHot for Department
    # Salary has natural order: low < medium < high
    salary_categories = [['low', 'medium', 'high']]
    salary_encoder = OrdinalEncoder(categories=salary_categories, dtype=int)

    # Department is nominal - use OneHotEncoder
    department_encoder = OneHotEncoder(sparse_output=False, drop='first', dtype=int)

    # Fit encoders on training data only
    X_train = X_train.copy()
    X_val = X_val.copy()
    X_test = X_test.copy()

    X_train['salary'] = salary_encoder.fit_transform(X_train[['salary']])
    X_val['salary'] = salary_encoder.transform(X_val[['salary']])
    X_test['salary'] = salary_encoder.transform(X_test[['salary']])

    # OneHot encode department - FIT FIRST to get categories
    X_train_dept = department_encoder.fit_transform(X_train[['department']])
    dept_columns = [f'Dept_{cat}' for cat in department_encoder.categories_[0][1:]]
    X_val_dept = department_encoder.transform(X_val[['department']])
    X_test_dept = department_encoder.transform(X_test[['department']])

    # Add one-hot columns and drop original department
    for i, col in enumerate(dept_columns):
        X_train[col] = X_train_dept[:, i]
        X_val[col] = X_val_dept[:, i]
        X_test[col] = X_test_dept[:, i]

    X_train = X_train.drop('department', axis=1)
    X_val = X_val.drop('department', axis=1)
    X_test = X_test.drop('department', axis=1)

    # Update feature columns
    feature_cols_encoded = [c for c in X_train.columns]

    # Encode full X for cross-validation
    X_encoded = X.copy()
    X_encoded['salary'] = salary_encoder.transform(X_encoded[['salary']])
    X_encoded_dept = department_encoder.transform(X_encoded[['department']])
    for i, col in enumerate(dept_columns):
        X_encoded[col] = X_encoded_dept[:, i]
    X_encoded = X_encoded.drop('department', axis=1)

    # Calculating class weight for imbalance
    neg_count, pos_count = np.bincount(y_train)
    scale_pos_weight = neg_count / pos_count

    models = {}
    results = {}
    confusion_matrices = {}
    classification_reports = {}
    optimal_thresholds = {}
    training_times = {}

    print("\nTraining Random Forest (Baseline)...")
    start_time = time.time()
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'
    )
    rf.fit(X_train[feature_cols_encoded], y_train)
    training_times["random_forest"] = time.time() - start_time

    # Use VALIDATION set for threshold tuning
    rf_val_proba = rf.predict_proba(X_val[feature_cols_encoded])[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y_val, rf_val_proba)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    optimal_idx = np.argmax(f1_scores)
    optimal_thresholds["random_forest"] = float(thresholds[optimal_idx])

    # Evaluate on TEST set with optimal threshold
    rf_test_proba = rf.predict_proba(X_test[feature_cols_encoded])[:, 1]
    rf_preds = (rf_test_proba >= optimal_thresholds["random_forest"]).astype(int)

    confusion_matrices["random_forest"] = confusion_matrix(y_test, rf_preds).tolist()
    classification_reports["random_forest"] = classification_report(y_test, rf_preds, output_dict=True)

    results["random_forest"] = {
        "accuracy": float((rf_preds == y_test).mean()),
        "roc_auc": float(roc_auc_score(y_test, rf_test_proba)),
        "cv_scores": cross_val_score(rf, X_encoded, y, cv=5, scoring="roc_auc").tolist(),
        "optimal_threshold": optimal_thresholds["random_forest"]
    }

    models["random_forest"] = rf

    print("\nTraining XGBoost...")
    start_time = time.time()
    xgb = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss',
        n_jobs=-1
    )
    xgb.fit(X_train[feature_cols_encoded], y_train)
    training_times["xgboost"] = time.time() - start_time

    # Use VALIDATION set for threshold tuning
    xgb_val_proba = xgb.predict_proba(X_val[feature_cols_encoded])[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y_val, xgb_val_proba)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    optimal_idx = np.argmax(f1_scores)
    optimal_thresholds["xgboost"] = float(thresholds[optimal_idx])

    # Evaluate on TEST set with optimal threshold
    xgb_test_proba = xgb.predict_proba(X_test[feature_cols_encoded])[:, 1]
    xgb_preds = (xgb_test_proba >= optimal_thresholds["xgboost"]).astype(int)

    confusion_matrices["xgboost"] = confusion_matrix(y_test, xgb_preds).tolist()
    classification_reports["xgboost"] = classification_report(y_test, xgb_preds, output_dict=True)

    results["xgboost"] = {
        "accuracy": float((xgb_preds == y_test).mean()),
        "roc_auc": float(roc_auc_score(y_test, xgb_test_proba)),
        "cv_scores": cross_val_score(xgb, X_encoded, y, cv=5, scoring="roc_auc").tolist(),
        "optimal_threshold": optimal_thresholds["xgboost"]
    }

    models["xgboost"] = xgb

    print("\nTraining LightGBM...")
    start_time = time.time()
    lgb = LGBMClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        class_weight='balanced',
        random_state=42,
        verbose=-1,
        n_jobs=-1
    )
    lgb.fit(X_train[feature_cols_encoded], y_train)
    training_times["lightgbm"] = time.time() - start_time

    # Use VALIDATION set for threshold tuning
    lgb_val_proba = lgb.predict_proba(X_val[feature_cols_encoded])[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y_val, lgb_val_proba)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    optimal_idx = np.argmax(f1_scores)
    optimal_thresholds["lightgbm"] = float(thresholds[optimal_idx])

    # Evaluate on TEST set with optimal threshold
    lgb_test_proba = lgb.predict_proba(X_test[feature_cols_encoded])[:, 1]
    lgb_preds = (lgb_test_proba >= optimal_thresholds["lightgbm"]).astype(int)

    confusion_matrices["lightgbm"] = confusion_matrix(y_test, lgb_preds).tolist()
    classification_reports["lightgbm"] = classification_report(y_test, lgb_preds, output_dict=True)

    results["lightgbm"] = {
        "accuracy": float((lgb_preds == y_test).mean()),
        "roc_auc": float(roc_auc_score(y_test, lgb_test_proba)),
        "cv_scores": cross_val_score(lgb, X_encoded, y, cv=5, scoring="roc_auc").tolist(),
        "optimal_threshold": optimal_thresholds["lightgbm"]
    }

    models["lightgbm"] = lgb

    return (models, results, confusion_matrices, classification_reports,
            optimal_thresholds, training_times, salary_encoder, department_encoder,
            feature_cols_encoded, X_train, X_val, X_test, y_train, y_val, y_test)


def compute_fairness_metrics(y_true, y_pred, y_proba, sensitive_features, feature_name):
    """Computing fairness metrics using fairlearn."""
    metrics = {
        'selection_rate': selection_rate,
        'accuracy': accuracy_score,
        'precision': precision_score,
        'recall': recall_score,
        'f1': f1_score,
        'true_positive_rate': true_positive_rate,
        'false_positive_rate': false_positive_rate
    }

    mf = MetricFrame(
        metrics=metrics,
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=sensitive_features
    )

    dp_diff = demographic_parity_difference(
        y_true=y_true, y_pred=y_pred, sensitive_features=sensitive_features
    )
    dp_ratio = demographic_parity_ratio(
        y_true=y_true, y_pred=y_pred, sensitive_features=sensitive_features
    )

    eod_diff = equalized_odds_difference(
        y_true=y_true, y_pred=y_pred, sensitive_features=sensitive_features
    )

    return {
        'by_group': mf.by_group.to_dict(),
        'overall': mf.overall.to_dict(),
        'demographic_parity_difference': float(dp_diff),
        'demographic_parity_ratio': float(dp_ratio),
        'passes_80_percent_rule': bool(dp_ratio >= 0.8),
        'equalized_odds_difference': float(eod_diff)
    }


def apply_threshold_optimizer(base_model, X_train, y_train, sensitive_features, prefit=True):
    """
    Applying post-processing mitigation using ThresholdOptimizer with equalized odds.

    Args:
        base_model: Pre-trained model to mitigate
        X_train: Training features
        y_train: Training labels
        sensitive_features: Sensitive attribute values for training data
        prefit: Whether base_model is already fitted (default: True)

    Returns:
        ThresholdOptimizer: Fitted mitigated model
    """
    print("\nApplying ThresholdOptimizer (equalized odds constraint)...")
    to = ThresholdOptimizer(
        estimator=base_model,
        constraints="equalized_odds",
        prefit=prefit
    )
    to.fit(X_train, y_train, sensitive_features=sensitive_features)
    print("  ThresholdOptimizer fitted successfully")
    return to


def apply_exponentiated_gradient(X_train, y_train, sensitive_features, estimator=None, eps=0.01):
    """
    Applying in-training mitigation using ExponentiatedGradient with equalized odds.

    Args:
        X_train: Training features
        y_train: Training labels
        sensitive_features: Sensitive attribute values for training data
        estimator: Base estimator (default: LightGBM)
        eps: Tolerance for constraint violation (default: 0.01)

    Returns:
        ExponentiatedGradient: Fitted mitigated model
    """
    print("\nApplying ExponentiatedGradient (equalized odds constraint)...")
    if estimator is None:
        estimator = LGBMClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            class_weight='balanced',
            random_state=42,
            verbose=-1,
            n_jobs=-1
        )

    constraint = EqualizedOdds()
    mitigated = ExponentiatedGradient(estimator, constraint, eps=eps)
    mitigated.fit(X_train, y_train, sensitive_features=sensitive_features)
    print("  ExponentiatedGradient fitted successfully")
    return mitigated


def evaluate_mitigated_model(mitigated_model, X_test, y_test, sensitive_features,
                             model_name="mitigated"):
    """
    Evaluating a mitigated model and computing fairness metrics.

    Args:
        mitigated_model: Fitted mitigated model
        X_test: Test features
        y_test: Test labels
        sensitive_features: Sensitive attribute values for test data
        model_name: Name of the model for reporting

    Returns:
        dict: Results including predictions, metrics, and fairness
    """
    y_test_array = y_test.values if hasattr(y_test, 'values') else y_test

    # Get predictions - ThresholdOptimizer requires sensitive_features
    # ExponentiatedGradient also requires sensitive_features
    try:
        y_pred = mitigated_model.predict(X_test, sensitive_features=sensitive_features)
    except TypeError:
        # Fallback for models that don't require sensitive_features
        y_pred = mitigated_model.predict(X_test)

    # Get probabilities based on model type
    if hasattr(mitigated_model, '_pmf_predict'):
        # ThresholdOptimizer
        try:
            pmf = mitigated_model._pmf_predict(X_test, sensitive_features=sensitive_features)
            y_proba = pmf[:, 1]  # Probability of positive class
        except TypeError:
            pmf = mitigated_model._pmf_predict(X_test)
            y_proba = pmf[:, 1]
    elif hasattr(mitigated_model, 'predict_proba'):
        # ExponentiatedGradient and others
        try:
            y_proba = mitigated_model.predict_proba(X_test, sensitive_features=sensitive_features)[:, 1]
        except TypeError:
            y_proba = mitigated_model.predict_proba(X_test)[:, 1]
    else:
        y_proba = y_pred.astype(float)

    # Compute standard metrics
    accuracy = accuracy_score(y_test_array, y_pred)
    precision = precision_score(y_test_array, y_pred, zero_division=0)
    recall = recall_score(y_test_array, y_pred, zero_division=0)
    f1 = f1_score(y_test_array, y_pred, zero_division=0)

    # Compute fairness metrics
    fairness = compute_fairness_metrics(
        y_test_array, y_pred, y_proba,
        sensitive_features=sensitive_features,
        feature_name='group'
    )

    return {
        'model_name': model_name,
        'predictions': y_pred,
        'probabilities': y_proba,
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'fairness': fairness
    }


def run_fairness_analysis(models, X_test, y_test, feature_cols_encoded):
    """Running fairness analysis on all models."""
    y_test_array = y_test.values if hasattr(y_test, 'values') else y_test

    # Reconstruct sensitive features from X_test
    salary_sensitive = X_test['salary'].map({0: 'low', 1: 'medium', 2: 'high'})

    dept_columns = [c for c in X_test.columns if c.startswith('Dept_')]
    dept_sensitive = 'unknown'
    for col in dept_columns:
        mask = X_test[col] == 1
        dept_sensitive = np.where(mask, col.replace('Dept_', ''), dept_sensitive)

    fairness_by_model = {}
    for model_name, model in models.items():
        y_proba = model.predict_proba(X_test)[:, 1]
        y_pred = (y_proba >= 0.5).astype(int)

        salary_fairness = compute_fairness_metrics(
            y_test_array, y_pred, y_proba,
            sensitive_features=salary_sensitive,
            feature_name='salary'
        )

        dept_fairness = compute_fairness_metrics(
            y_test_array, y_pred, y_proba,
            sensitive_features=dept_sensitive,
            feature_name='department'
        )

        fairness_by_model[model_name] = {
            'by_salary': salary_fairness,
            'by_department': dept_fairness
        }

    return fairness_by_model


def run_mitigation_analysis(base_model, X_train, y_train, X_test, y_test,
                            salary_sensitive_train, salary_sensitive_test,
                            dept_sensitive_train, dept_sensitive_test,
                            feature_cols_encoded, model_name="base", is_xgboost=False):
    """
    Running mitigation analysis using ThresholdOptimizer and ExponentiatedGradient.

    Args:
        base_model: Pre-trained baseline model
        X_train, y_train: Training data
        X_test, y_test: Test data
        salary_sensitive_train, salary_sensitive_test: Salary sensitive features
        dept_sensitive_train, dept_sensitive_test: Department sensitive features
        feature_cols_encoded: List of encoded feature columns
        model_name: Name of the base model

    Returns:
        dict: Mitigation results for both techniques
    """
    print(f"\n{'='*70}")
    print(f"MITIGATION ANALYSIS FOR {model_name.upper()}")
    print('='*70)

    results = {}

    # Wrap XGBoost models for fairlearn compatibility
    if is_xgboost:
        # Get XGBoost parameters from the base model
        xgb_params = base_model.get_params()
        # Create wrapped version with same parameters
        wrapped_model = Float64XGBoostWrapper(**xgb_params)
        # Fit the wrapper on the same data
        wrapped_model.fit(X_train[feature_cols_encoded].values, y_train)
        base_model = wrapped_model
        print("Using Float64XGBoostWrapper for fairlearn compatibility")

    def get_values(obj):
        if hasattr(obj, 'values'):
            return obj.values
        return obj

    # 1. ThresholdOptimizer with salary constraint
    print("\n--- ThresholdOptimizer (Salary Equalized Odds) ---")
    to_salary = apply_threshold_optimizer(
        base_model, X_train[feature_cols_encoded].values,
        get_values(y_train), get_values(salary_sensitive_train)
    )
    to_salary_results = evaluate_mitigated_model(
        to_salary, X_test[feature_cols_encoded].values,
        get_values(y_test), get_values(salary_sensitive_test),
        model_name=f"{model_name}_to_salary"
    )
    results['threshold_optimizer_salary'] = to_salary_results

    # 2. ThresholdOptimizer with department constraint
    print("\n--- ThresholdOptimizer (Department Equalized Odds) ---")
    to_dept = apply_threshold_optimizer(
        base_model, X_train[feature_cols_encoded].values,
        get_values(y_train), get_values(dept_sensitive_train)
    )
    to_dept_results = evaluate_mitigated_model(
        to_dept, X_test[feature_cols_encoded].values,
        get_values(y_test), get_values(dept_sensitive_test),
        model_name=f"{model_name}_to_dept"
    )
    results['threshold_optimizer_dept'] = to_dept_results

    # 3. ExponentiatedGradient with salary constraint
    print("\n--- ExponentiatedGradient (Salary Equalized Odds) ---")
    eg_salary = apply_exponentiated_gradient(
        X_train[feature_cols_encoded].values,
        get_values(y_train), get_values(salary_sensitive_train)
    )
    eg_salary_results = evaluate_mitigated_model(
        eg_salary, X_test[feature_cols_encoded].values,
        get_values(y_test), get_values(salary_sensitive_test),
        model_name=f"{model_name}_eg_salary"
    )
    results['exponentiated_gradient_salary'] = eg_salary_results

    return results


def extract_feature_importance(models, feature_cols):
    """Extracting and formatting feature importance from all models."""
    importances = {}

    for name, model in models.items():
        importance_df = pd.DataFrame({
            "feature": feature_cols,
            "importance": model.feature_importances_
        }).sort_values("importance", ascending=False)
        importances[name] = importance_df

    return importances


def save_artifacts(models, results, feature_importances, salary_encoder,
                   department_encoder, feature_cols, confusion_matrices,
                   classification_reports, optimal_thresholds, training_times,
                   fairness_results=None, mitigated_models=None,
                   mitigation_results=None):
    """Saving model artifacts for use in dashboard."""
    output_dir = Path(os.getenv("MODELS_DIR", "models"))
    output_dir.mkdir(exist_ok=True)

    joblib.dump(models["random_forest"], output_dir / "random_forest_model.pkl")
    joblib.dump(models["xgboost"], output_dir / "xgboost_model.pkl")
    joblib.dump(models["lightgbm"], output_dir / "lightgbm_model.pkl")

    joblib.dump(salary_encoder, output_dir / "salary_encoder.pkl")
    joblib.dump(department_encoder, output_dir / "department_encoder.pkl")

    # Save preprocessing artifacts as a single file for faster loading
    # Note: scaler is None for tree-based models (no scaling needed)
    save_preprocessing_artifacts(
        scaler=None,
        salary_encoder=salary_encoder,
        department_encoder=department_encoder,
        feature_cols=feature_cols,
        output_dir=output_dir
    )

    for model_name in results:
        results[model_name]["training_time"] = training_times[model_name]

    with open(output_dir / "model_results.json", "w") as f:
        json.dump(results, f, indent=2)

    with open(output_dir / "confusion_matrix.json", "w") as f:
        json.dump(confusion_matrices, f, indent=2)

    json_reports = {}
    for model_name, report in classification_reports.items():
        json_reports[model_name] = {}
        for key, value in report.items():
            if isinstance(value, dict):
                json_reports[model_name][key] = {
                    k: float(v) if isinstance(v, (np.integer, np.floating)) else v
                    for k, v in value.items()
                }
            else:
                json_reports[model_name][key] = float(value) if isinstance(value, (np.integer, np.floating)) else value

    with open(output_dir / "classification_report.json", "w") as f:
        json.dump(json_reports, f, indent=2)

    with open(output_dir / "optimal_threshold.json", "w") as f:
        json.dump(optimal_thresholds, f, indent=2)

    for name, importance_df in feature_importances.items():
        importance_df.to_csv(output_dir / f"feature_importance_{name}.csv", index=False)

    comparison_data = []
    for name, importance_df in feature_importances.items():
        top_features = importance_df.head(3)
        comparison_data.append({
            "model": name,
            "top_feature": top_features.iloc[0]["feature"],
            "top_importance": float(top_features.iloc[0]["importance"])
        })

    pd.DataFrame(comparison_data).to_csv(output_dir / "model_comparison.csv", index=False)

    if fairness_results:
        with open(output_dir / "fairness_report.json", "w") as f:
            json.dump(fairness_results, f, indent=2)
        print(f"\nFairness report saved")

    if mitigated_models:
        mitigation_dir = output_dir / "mitigated"
        mitigation_dir.mkdir(exist_ok=True)
        for model_name, model in mitigated_models.items():
            joblib.dump(model, mitigation_dir / f"{model_name}.pkl")
        print(f"Mitigated models saved to {mitigation_dir}/")

    if mitigation_results:
        def convert_mitigation(obj):
            if isinstance(obj, dict):
                return {k: convert_mitigation(v) for k, v in obj.items()}
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, pd.DataFrame):
                return obj.to_dict()
            elif pd.isna(obj):
                return None
            return obj

        converted_results = convert_mitigation(mitigation_results)
        with open(output_dir / "mitigation_report.json", "w") as f:
            json.dump(converted_results, f, indent=2)
        print(f"Mitigation report saved")

    print(f"\nModel artifacts saved to {output_dir}/")


def print_summary(results, feature_importances, training_times):
    """Printing model performance summary."""
    print("\n" + "=" * 70)
    print("MODEL PERFORMANCE SUMMARY")
    print("=" * 70)

    for model_name in ["random_forest", "xgboost", "lightgbm"]:
        metrics = results[model_name]
        display_name = model_name.replace("_", " ").title()
        print(f"\n{display_name}:")
        print(f"  Accuracy:       {metrics['accuracy']:.4f} ({metrics['accuracy'] * 100:.2f}%)")
        print(f"  ROC AUC:        {metrics['roc_auc']:.4f}")
        print(f"  CV Scores:      {np.array(metrics['cv_scores']).mean():.4f} (+/- {np.array(metrics['cv_scores']).std() * 2:.4f})")
        print(f"  Training Time:  {training_times[model_name]:.2f}s")
        print(f"  Opt Threshold:  {metrics['optimal_threshold']:.4f}")

    print("\n" + "=" * 70)
    print("FEATURE IMPORTANCE COMPARISON (Top 3)")
    print("=" * 70)

    for model_name in ["random_forest", "xgboost", "lightgbm"]:
        display_name = model_name.replace("_", " ").title()
        print(f"\n{display_name}:")
        top_features = feature_importances[model_name].head(3)
        for _, row in top_features.iterrows():
            print(f"  {row['feature']}: {row['importance']:.4f}")


def print_comparison_report(results):
    """Printing detailed comparison report."""
    print("\n" + "=" * 70)
    print("MODEL COMPARISON REPORT")
    print("=" * 70)

    # Finding best model
    best_accuracy = max(results.items(), key=lambda x: x[1]["accuracy"])
    best_roc_auc = max(results.items(), key=lambda x: x[1]["roc_auc"])

    print(f"\nBest Accuracy: {best_accuracy[0].replace('_', ' ').title()} ({best_accuracy[1]['accuracy'] * 100:.2f}%)")
    print(f"Best ROC AUC:  {best_roc_auc[0].replace('_', ' ').title()} ({best_roc_auc[1]['roc_auc']:.4f})")

    # Baseline comparison (reference)
    rf_accuracy = results["random_forest"]["accuracy"]
    rf_roc_auc = results["random_forest"]["roc_auc"]

    print(f"\nBaseline (Random Forest): {rf_accuracy * 100:.2f}% accuracy, {rf_roc_auc:.4f} ROC AUC")

    for model_name in ["xgboost", "lightgbm"]:
        metrics = results[model_name]
        acc_delta = (metrics["accuracy"] - rf_accuracy) * 100
        auc_delta = metrics["roc_auc"] - rf_roc_auc

        status = ""
        if metrics["accuracy"] >= rf_accuracy and metrics["roc_auc"] >= rf_roc_auc:
            status = "✓ Meets or exceeds baseline"
        elif metrics["accuracy"] >= 0.95:
            status = "~ Meets minimum threshold (95%)"
        else:
            status = "✗ Below threshold"

        print(f"\n{model_name.replace('_', ' ').title()}:")
        print(f"  Accuracy: {metrics['accuracy'] * 100:.2f}% ({acc_delta:+.2f}%)")
        print(f"  ROC AUC:  {metrics['roc_auc']:.4f} ({auc_delta:+.4f})")
        print(f"  Status:   {status}")


def main():
    parser = argparse.ArgumentParser(description='Train advanced ML models with optional fairness mitigation')
    parser.add_argument('--mitigate', action='store_true',
                        help='Apply fairness mitigation techniques (Equalized Odds)')
    args = parser.parse_args()

    print("=" * 70)
    print("ADVANCED EMPLOYEE ATTRITION PREDICTION MODELS")
    print("XGBoost & LightGBM vs Random Forest Baseline")
    if args.mitigate:
        print("WITH EQUALIZED ODDS MITIGATION")
    print("=" * 70)

    # Loading data
    print("\nLoading data...")
    df, X, y, feature_cols = load_and_preprocess_data()
    print(f"Loaded {len(X)} samples with {len(feature_cols)} features")
    print(f"Attrition rate: {y.mean() * 100:.1f}%")

    # Training models
    (models, results, confusion_matrices, classification_reports,
     optimal_thresholds, training_times, salary_enc, dept_enc,
     feature_cols_encoded, X_train, X_val, X_test, y_train, y_val, y_test) = train_models(df, X, y, feature_cols)

    # Extracting feature importance
    feature_importances = extract_feature_importance(models, feature_cols_encoded)

    # Run fairness analysis
    print("\nRunning fairness analysis...")
    fairness_results = run_fairness_analysis(models, X_test, y_test, feature_cols_encoded)

    # Print fairness summary
    print("\n" + "=" * 70)
    print("BASELINE FAIRNESS SUMMARY")
    print("=" * 70)
    salary_ratio = fairness_results['random_forest']['by_salary']['demographic_parity_ratio']
    dept_ratio = fairness_results['random_forest']['by_department']['demographic_parity_ratio']
    eod_salary = fairness_results['random_forest']['by_salary'].get('equalized_odds_difference', 0)
    eod_dept = fairness_results['random_forest']['by_department'].get('equalized_odds_difference', 0)

    print(f"\nSalary Demographic Parity Ratio: {salary_ratio:.4f}")
    print(f"  {'✓ Passes' if salary_ratio >= 0.8 else '⚠️ Fails'} EEOC 80% rule")
    print(f"  Equalized Odds Difference: {eod_salary:.4f}")
    print(f"\nDepartment Demographic Parity Ratio: {dept_ratio:.4f}")
    print(f"  {'✓ Passes' if dept_ratio >= 0.8 else '⚠️ Fails'} EEOC 80% rule")
    print(f"  Equalized Odds Difference: {eod_dept:.4f}")

    # Run mitigation if requested
    mitigated_models = {}
    mitigation_results = {}

    if args.mitigate:
        print("\n" + "=" * 70)
        print("RUNNING FAIRNESS MITIGATION")
        print("=" * 70)

        # Create sensitive features from encoded data
        salary_sensitive_train = X_train['salary'].map({0: 'low', 1: 'medium', 2: 'high'})
        salary_sensitive_test = X_test['salary'].map({0: 'low', 1: 'medium', 2: 'high'})

        dept_columns = [c for c in X_test.columns if c.startswith('Dept_')]
        dept_sensitive_train = 'unknown'
        for col in dept_columns:
            mask = X_train[col] == 1
            dept_sensitive_train = np.where(mask, col.replace('Dept_', ''), dept_sensitive_train)
        dept_sensitive_test = 'unknown'
        for col in dept_columns:
            mask = X_test[col] == 1
            dept_sensitive_test = np.where(mask, col.replace('Dept_', ''), dept_sensitive_test)

        # Applying Equalized Odds mitigation to both XGBoost and LightGBM
        mitigation_results = {}

        for model_name in ["xgboost", "lightgbm"]:
            base_model = models[model_name]
            print(f"\n{'='*70}")
            print(f"Processing {model_name.upper()} for Equalized Odds mitigation")
            print('='*70)

            model_mitigation = run_mitigation_analysis(
                base_model, X_train, y_train, X_test, y_test,
                salary_sensitive_train, salary_sensitive_test,
                dept_sensitive_train, dept_sensitive_test,
                feature_cols_encoded, model_name, is_xgboost=(model_name == "xgboost")
            )
            mitigation_results[model_name] = model_mitigation

            # Collect mitigated models for saving
            for key, result in model_mitigation.items():
                mitigated_models[f"{model_name}_{key}"] = result['predictions']

        # Print mitigation summary
        print("\n" + "=" * 70)
        print("MITIGATION RESULTS SUMMARY")
        print("=" * 70)

        for model_name in ["xgboost", "lightgbm"]:
            baseline_acc = results[model_name]['accuracy']
            baseline_eod = fairness_results[model_name]['by_salary'].get('equalized_odds_difference', 0)

            print(f"\nBaseline {model_name.upper()}:")
            print(f"  Accuracy: {baseline_acc:.4f}")
            print(f"  Equalized Odds Difference: {baseline_eod:.4f}")

            for tech_name, tech_results in mitigation_results[model_name].items():
                acc = tech_results['accuracy']
                eod = tech_results['fairness']['equalized_odds_difference']
                acc_delta = acc - baseline_acc
                eod_delta = eod - baseline_eod

                print(f"\n{tech_name.upper().replace('_', ' ')}:")
                print(f"  Accuracy: {acc:.4f} ({acc_delta:+.4f})")
                print(f"  Equalized Odds Difference: {eod:.4f} ({eod_delta:+.4f})")

    # Saving everything
    save_artifacts(models, results, feature_importances, salary_enc, dept_enc,
                   feature_cols_encoded, confusion_matrices, classification_reports,
                   optimal_thresholds, training_times, fairness_results,
                   mitigated_models if args.mitigate else None,
                   mitigation_results if args.mitigate else None)

    # Printing summaries
    print_summary(results, feature_importances, training_times)
    print_comparison_report(results)


if __name__ == "__main__":
    main()
