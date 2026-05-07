#!/usr/bin/env python3
"""
Train Employee Attrition Prediction Model
Creates and evaluates classification model, saves model and insights.

BIAS FIXES APPLIED:
- Proper ordinal encoding for salary (low=0, medium=1, high=2)
- OneHot encoding for Department (nominal variable)
- Train/Val/Test split (70/15/15) to prevent test set leakage
- Threshold tuning on validation set only
- Model calibration for reliable probability outputs
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import json

from sklearn.model_selection import train_test_split, cross_val_score
from config import HR_DATA_PATH, MODELS_DIR
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    precision_recall_curve, brier_score_loss, accuracy_score,
    precision_score, recall_score, f1_score
)

# Fairness analysis
from fairlearn.metrics import (
    MetricFrame,
    demographic_parity_difference,
    demographic_parity_ratio,
    selection_rate
)


def bootstrap_metric_ci(y_true, y_pred, y_proba, metric_func, n_bootstrap=1000, ci=0.95):
    """Compute bootstrap confidence intervals for a metric."""
    np.random.seed(42)
    n = len(y_true)
    bootstrap_scores = []

    for _ in range(n_bootstrap):
        indices = np.random.choice(n, n, replace=True)
        y_true_boot = y_true[indices]
        y_pred_boot = y_pred[indices] if y_pred is not None else None
        y_proba_boot = y_proba[indices] if y_proba is not None else None

        if metric_func == 'accuracy':
            score = accuracy_score(y_true_boot, y_pred_boot)
        elif metric_func == 'roc_auc':
            score = roc_auc_score(y_true_boot, y_proba_boot)
        else:
            score = metric_func(y_true_boot, y_proba_boot)

        bootstrap_scores.append(score)

    alpha = 1 - ci
    lower = np.percentile(bootstrap_scores, 100 * alpha / 2)
    upper = np.percentile(bootstrap_scores, 100 * (1 - alpha / 2))
    mean = np.mean(bootstrap_scores)
    std = np.std(bootstrap_scores)

    return {"mean": float(mean), "std": float(std), "lower": float(lower), "upper": float(upper)}


def load_and_preprocess_data(csv_path: str = None):
    """Load and preprocess data for modeling.

    Args:
        csv_path: Path to CSV file (defaults to config.HR_DATA_PATH)
    """
    if csv_path is None:
        csv_path = HR_DATA_PATH
    df = pd.read_csv(csv_path)

    # Drop Emp_Id (not predictive)
    df = df.drop("Emp_Id", axis=1)

    # Feature columns
    feature_cols = [c for c in df.columns if c != "left"]
    X = df[feature_cols]
    y = df["left"]

    return df, X, y, feature_cols


def train_models(df, X, y):
    """Train multiple models and compare performance."""
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

    # OneHot encode Department - FIT FIRST to get categories
    X_train_dept = department_encoder.fit_transform(X_train[['Department']])
    dept_columns = [f'Dept_{cat}' for cat in department_encoder.categories_[0][1:]]
    X_val_dept = department_encoder.transform(X_val[['Department']])
    X_test_dept = department_encoder.transform(X_test[['Department']])

    # Add one-hot columns and drop original Department
    for i, col in enumerate(dept_columns):
        X_train[col] = X_train_dept[:, i]
        X_val[col] = X_val_dept[:, i]
        X_test[col] = X_test_dept[:, i]

    X_train = X_train.drop('Department', axis=1)
    X_val = X_val.drop('Department', axis=1)
    X_test = X_test.drop('Department', axis=1)

    # Update feature columns
    feature_cols_encoded = [c for c in X_train.columns if c != 'left']

    # Encode full X for cross-validation
    X_encoded = X.copy()
    X_encoded['salary'] = salary_encoder.transform(X_encoded[['salary']])
    X_encoded_dept = department_encoder.transform(X_encoded[['Department']])
    for i, col in enumerate(dept_columns):
        X_encoded[col] = X_encoded_dept[:, i]
    X_encoded = X_encoded.drop('Department', axis=1)

    # Scale features for Logistic Regression
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train[feature_cols_encoded])
    X_val_scaled = scaler.transform(X_val[feature_cols_encoded])
    X_test_scaled = scaler.transform(X_test[feature_cols_encoded])

    models = {}
    results = {}
    confusion_matrices = {}
    classification_reports = {}
    optimal_thresholds = {}

    # Random Forest
    print("\nTraining Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'
    )
    rf.fit(X_train[feature_cols_encoded], y_train)

    # Use VALIDATION set for threshold tuning (prevents test set leakage)
    rf_val_proba = rf.predict_proba(X_val[feature_cols_encoded])[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y_val, rf_val_proba)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    optimal_idx = np.argmax(f1_scores)
    optimal_thresholds["random_forest"] = float(thresholds[optimal_idx])
    optimal_threshold = optimal_thresholds["random_forest"]

    # Evaluate on TEST set with optimal threshold
    rf_test_proba = rf.predict_proba(X_test[feature_cols_encoded])[:, 1]
    rf_preds = (rf_test_proba >= optimal_threshold).astype(int)

    # Confusion Matrix
    cm_rf = confusion_matrix(y_test, rf_preds)
    confusion_matrices["random_forest"] = cm_rf.tolist()

    # Classification Report
    cr_rf = classification_report(y_test, rf_preds, output_dict=True)
    classification_reports["random_forest"] = cr_rf

    models["random_forest"] = {
        "model": rf,
        "scaler": None,
        "encoder": department_encoder,
        "predictions": rf_preds,
        "probabilities": rf_test_proba
    }

    results["random_forest"] = {
        "accuracy": (rf_preds == y_test).mean(),
        "roc_auc": roc_auc_score(y_test, rf_test_proba),
        "cv_scores": cross_val_score(rf, X_encoded, y, cv=5, scoring="roc_auc").tolist(),
        "optimal_threshold": optimal_threshold
    }

    # Logistic Regression
    print("Training Logistic Regression...")
    lr = LogisticRegression(
        random_state=42,
        max_iter=1000,
        class_weight='balanced'
    )
    lr.fit(X_train_scaled, y_train)

    # Use VALIDATION set for threshold tuning
    lr_val_proba = lr.predict_proba(X_val_scaled)[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y_val, lr_val_proba)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    optimal_idx = np.argmax(f1_scores)
    optimal_thresholds["logistic_regression"] = float(thresholds[optimal_idx])
    optimal_threshold_lr = optimal_thresholds["logistic_regression"]

    # Evaluate on TEST set with optimal threshold
    lr_test_proba = lr.predict_proba(X_test_scaled)[:, 1]
    lr_preds = (lr_test_proba >= optimal_threshold_lr).astype(int)

    # Confusion Matrix
    cm_lr = confusion_matrix(y_test, lr_preds)
    confusion_matrices["logistic_regression"] = cm_lr.tolist()

    # Classification Report
    cr_lr = classification_report(y_test, lr_preds, output_dict=True)
    classification_reports["logistic_regression"] = cr_lr

    models["logistic_regression"] = {
        "model": lr,
        "scaler": scaler,
        "encoder": department_encoder,
        "predictions": lr_preds,
        "probabilities": lr_test_proba
    }

    # CV for LR requires scaling - use pipeline
    lr_pipeline = make_pipeline(
        StandardScaler(),
        LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')
    )
    results["logistic_regression"] = {
        "accuracy": (lr_preds == y_test).mean(),
        "roc_auc": roc_auc_score(y_test, lr_test_proba),
        "cv_scores": cross_val_score(lr_pipeline, X_encoded, y, cv=5, scoring="roc_auc").tolist(),
        "optimal_threshold": optimal_threshold_lr
    }

    # MODEL CALIBRATION for reliable probability outputs
    print("\nCalibrating models for reliable probability outputs...")

    # Calibrate Random Forest using validation set
    rf_calibrated = CalibratedClassifierCV(rf, method='isotonic', ensemble=False)
    rf_calibrated.fit(X_val[feature_cols_encoded], y_val)
    rf_calib_proba = rf_calibrated.predict_proba(X_test[feature_cols_encoded])[:, 1]
    rf_brier_uncalib = brier_score_loss(y_test, rf_test_proba)
    rf_brier_calib = brier_score_loss(y_test, rf_calib_proba)

    print(f"  RF - Brier Score (uncalibrated): {rf_brier_uncalib:.4f}")
    print(f"  RF - Brier Score (calibrated):   {rf_brier_calib:.4f}")

    # Calibrate Logistic Regression using validation set
    lr_calibrated = CalibratedClassifierCV(lr, method='isotonic', ensemble=False)
    lr_calibrated.fit(X_val_scaled, y_val)
    lr_calib_proba = lr_calibrated.predict_proba(X_test_scaled)[:, 1]
    lr_brier_uncalib = brier_score_loss(y_test, lr_test_proba)
    lr_brier_calib = brier_score_loss(y_test, lr_calib_proba)

    print(f"  LR - Brier Score (uncalibrated): {lr_brier_uncalib:.4f}")
    print(f"  LR - Brier Score (calibrated):   {lr_brier_calib:.4f}")

    # Add calibrated models to results
    results["random_forest"]["brier_score_uncalibrated"] = float(rf_brier_uncalib)
    results["random_forest"]["brier_score_calibrated"] = float(rf_brier_calib)
    results["logistic_regression"]["brier_score_uncalibrated"] = float(lr_brier_uncalib)
    results["logistic_regression"]["brier_score_calibrated"] = float(lr_brier_calib)

    # BOOTSTRAP CONFIDENCE INTERVALS
    print("\nComputing bootstrap confidence intervals (95% CI)...")

    y_test_array = y_test.values if hasattr(y_test, 'values') else y_test

    # RF confidence intervals
    rf_acc_ci = bootstrap_metric_ci(y_test_array, rf_preds, None, 'accuracy')
    rf_auc_ci = bootstrap_metric_ci(y_test_array, None, rf_test_proba, 'roc_auc')

    results["random_forest"]["accuracy_ci"] = rf_acc_ci
    results["random_forest"]["roc_auc_ci"] = rf_auc_ci

    print(f"  RF Accuracy:  {rf_acc_ci['mean']:.4f} ({rf_acc_ci['lower']:.4f} - {rf_acc_ci['upper']:.4f})")
    print(f"  RF ROC AUC:   {rf_auc_ci['mean']:.4f} ({rf_auc_ci['lower']:.4f} - {rf_auc_ci['upper']:.4f})")

    # LR confidence intervals
    lr_acc_ci = bootstrap_metric_ci(y_test_array, lr_preds, None, 'accuracy')
    lr_auc_ci = bootstrap_metric_ci(y_test_array, None, lr_test_proba, 'roc_auc')

    results["logistic_regression"]["accuracy_ci"] = lr_acc_ci
    results["logistic_regression"]["roc_auc_ci"] = lr_auc_ci

    print(f"  LR Accuracy:  {lr_acc_ci['mean']:.4f} ({lr_acc_ci['lower']:.4f} - {lr_acc_ci['upper']:.4f})")
    print(f"  LR ROC AUC:   {lr_auc_ci['mean']:.4f} ({lr_auc_ci['lower']:.4f} - {lr_auc_ci['upper']:.4f})")

    # Update models dict with calibrated versions
    models["random_forest_calibrated"] = {
        "model": rf_calibrated,
        "scaler": None,
        "encoder": department_encoder,
        "predictions": (rf_calib_proba >= optimal_threshold).astype(int),
        "probabilities": rf_calib_proba
    }

    models["logistic_regression_calibrated"] = {
        "model": lr_calibrated,
        "scaler": scaler,
        "encoder": department_encoder,
        "predictions": (lr_calib_proba >= optimal_threshold_lr).astype(int),
        "probabilities": lr_calib_proba
    }

    return (models, results, X_test[feature_cols_encoded], y_test,
            confusion_matrices, classification_reports, optimal_thresholds,
            salary_encoder, department_encoder, feature_cols_encoded)


def compute_fairness_metrics(y_true, y_pred, y_proba, sensitive_features, feature_name):
    """Compute fairness metrics using fairlearn."""
    metrics = {
        'selection_rate': selection_rate,
        'accuracy': accuracy_score,
        'precision': precision_score,
        'recall': recall_score,
        'f1': f1_score
    }

    mf = MetricFrame(
        metrics=metrics,
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=sensitive_features
    )

    # Demographic parity (EEOC 80% rule)
    dp_diff = demographic_parity_difference(
        y_true=y_true, y_pred=y_pred, sensitive_features=sensitive_features
    )
    dp_ratio = demographic_parity_ratio(
        y_true=y_true, y_pred=y_pred, sensitive_features=sensitive_features
    )

    return {
        'by_group': mf.by_group.to_dict(),
        'overall': mf.overall.to_dict(),
        'demographic_parity_difference': float(dp_diff),
        'demographic_parity_ratio': float(dp_ratio),
        'passes_80_percent_rule': bool(dp_ratio >= 0.8)
    }


def run_fairness_analysis(model, X_test, y_test, feature_cols_encoded):
    """Run fairness analysis on salary and department."""
    y_test_array = y_test.values if hasattr(y_test, 'values') else y_test
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    # Reconstruct sensitive features from X_test (salary is encoded, dept is one-hot)
    # We need to map back from encoded features to original categories
    # For simplicity, use the encoded salary directly (0=low, 1=medium, 2=high)
    salary_sensitive = X_test['salary'].map({0: 'low', 1: 'medium', 2: 'high'})

    # For department, find the one-hot column with value 1 (none means dropped baseline)
    dept_columns = [c for c in X_test.columns if c.startswith('Dept_')]
    dept_sensitive = 'unknown'
    for col in dept_columns:
        mask = X_test[col] == 1
        dept_sensitive = np.where(mask, col.replace('Dept_', ''), dept_sensitive)

    # Compute fairness metrics
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

    return {
        'by_salary': salary_fairness,
        'by_department': dept_fairness
    }


def extract_feature_importance(model, feature_cols):
    """Extract and format feature importance."""
    importances = pd.DataFrame({
        "feature": feature_cols,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)

    return importances


def save_artifacts(models, results, feature_importance, salary_encoder,
                   department_encoder, feature_cols, confusion_matrices,
                   classification_reports, optimal_thresholds, fairness_results=None):
    """Save model artifacts for use in dashboard."""
    output_dir = MODELS_DIR
    output_dir.mkdir(exist_ok=True)

    # Save models
    joblib.dump(models["random_forest"]["model"], output_dir / "rf_attrition_model.pkl")
    joblib.dump(models["random_forest_calibrated"]["model"], output_dir / "rf_attrition_model_calibrated.pkl")
    joblib.dump(models["logistic_regression"]["model"], output_dir / "lr_attrition_model.pkl")
    joblib.dump(models["logistic_regression_calibrated"]["model"], output_dir / "lr_attrition_model_calibrated.pkl")
    joblib.dump(models["logistic_regression"]["scaler"], output_dir / "scaler.pkl")

    # Save encoders
    joblib.dump(salary_encoder, output_dir / "salary_encoder.pkl")
    joblib.dump(department_encoder, output_dir / "department_encoder.pkl")

    # Save feature names
    with open(output_dir / "feature_columns.json", "w") as f:
        json.dump(feature_cols, f)

    # Save results
    with open(output_dir / "model_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Save confusion matrices
    with open(output_dir / "confusion_matrix.json", "w") as f:
        json.dump(confusion_matrices, f, indent=2)

    # Save classification reports
    # Convert numpy types to JSON-serializable format
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

    # Save optimal thresholds
    with open(output_dir / "optimal_threshold.json", "w") as f:
        json.dump(optimal_thresholds, f, indent=2)

    # Save feature importance
    feature_importance.to_csv(output_dir / "feature_importance.csv", index=False)

    # Save fairness report if available
    if fairness_results:
        with open(output_dir / "fairness_report.json", "w") as f:
            json.dump(fairness_results, f, indent=2)
        print(f"\nFairness report saved")

    print(f"\nModel artifacts saved to {output_dir}/")


def print_summary(results, feature_importance):
    """Print model performance summary."""
    print("\n" + "=" * 60)
    print("MODEL PERFORMANCE SUMMARY")
    print("=" * 60)

    for model_name, metrics in results.items():
        print(f"\n{model_name.replace('_', ' ').title()}:")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        if 'accuracy_ci' in metrics:
            ci = metrics['accuracy_ci']
            print(f"    95% CI: [{ci['lower']:.4f}, {ci['upper']:.4f}]")
        print(f"  ROC AUC:   {metrics['roc_auc']:.4f}")
        if 'roc_auc_ci' in metrics:
            ci = metrics['roc_auc_ci']
            print(f"    95% CI: [{ci['lower']:.4f}, {ci['upper']:.4f}]")
        print(f"  CV Scores: {np.array(metrics['cv_scores']).mean():.4f} (+/- {np.array(metrics['cv_scores']).std() * 2:.4f})")
        print(f"  Optimal Threshold: {metrics['optimal_threshold']:.4f}")
        if 'brier_score_calibrated' in metrics:
            print(f"  Calibration (Brier): {metrics['brier_score_uncalibrated']:.4f} -> {metrics['brier_score_calibrated']:.4f}")

    print("\n" + "=" * 60)
    print("TOP FEATURE IMPORTANCE (Random Forest)")
    print("=" * 60)
    print(feature_importance.to_string(index=False))


def main():
    print("=" * 60)
    print("EMPLOYEE ATTRITION PREDICTION MODEL")
    print("=" * 60)

    # Load data
    print("\nLoading data...")
    df, X, y, feature_cols = load_and_preprocess_data()
    print(f"Loaded {len(X)} samples with {len(feature_cols)} features")
    print(f"Attrition rate: {y.mean() * 100:.1f}%")

    # Train models
    (models, results, X_test, y_test, confusion_matrices,
     classification_reports, optimal_thresholds, salary_enc,
     dept_enc, feature_cols_encoded) = train_models(df, X, y)

    # Extract feature importance from Random Forest
    feature_importance = extract_feature_importance(
        models["random_forest"]["model"],
        feature_cols_encoded
    )

    # Run fairness analysis
    print("\nRunning fairness analysis...")
    fairness_results = run_fairness_analysis(
        models["random_forest"]["model"],
        X_test,
        y_test,
        feature_cols_encoded
    )

    # Print fairness summary
    print("\n" + "=" * 60)
    print("FAIRNESS SUMMARY")
    print("=" * 60)
    salary_ratio = fairness_results['by_salary']['demographic_parity_ratio']
    dept_ratio = fairness_results['by_department']['demographic_parity_ratio']
    print(f"\nSalary Demographic Parity Ratio: {salary_ratio:.4f}")
    print(f"  {'[PASS]' if salary_ratio >= 0.8 else '[FAIL]'} EEOC 80% rule")
    print(f"\nDepartment Demographic Parity Ratio: {dept_ratio:.4f}")
    print(f"  {'[PASS]' if dept_ratio >= 0.8 else '[FAIL]'} EEOC 80% rule")

    # Save everything
    save_artifacts(models, results, feature_importance, salary_enc, dept_enc,
                   feature_cols_encoded, confusion_matrices,
                   classification_reports, optimal_thresholds, fairness_results)

    # Print summary
    print_summary(results, feature_importance)


if __name__ == "__main__":
    main()
