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
from dotenv import load_dotenv

# Loading environment variables
load_dotenv()

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# Fairness analysis
from fairlearn.metrics import (
    MetricFrame,
    demographic_parity_difference,
    demographic_parity_ratio,
    selection_rate
)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def load_and_preprocess_data(csv_path: str = None):
    """Loading and preprocessing data for modeling."""
    if csv_path is None:
        csv_path = os.getenv("HR_DATA_PATH", "data/hr_employee_data.csv")
    df = pd.read_csv(csv_path)

    # Dropping Emp_Id (not predictive)
    df = df.drop("Emp_Id", axis=1)

    # Feature columns
    feature_cols = [c for c in df.columns if c != "left"]
    X = df[feature_cols]
    y = df["left"]

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
    feature_cols_encoded = [c for c in X_train.columns]

    # Encode full X for cross-validation
    X_encoded = X.copy()
    X_encoded['salary'] = salary_encoder.transform(X_encoded[['salary']])
    X_encoded_dept = department_encoder.transform(X_encoded[['Department']])
    for i, col in enumerate(dept_columns):
        X_encoded[col] = X_encoded_dept[:, i]
    X_encoded = X_encoded.drop('Department', axis=1)

    # Calculating class weight for imbalance
    neg_count, pos_count = np.bincount(y_train)
    scale_pos_weight = neg_count / pos_count

    models = {}
    results = {}
    confusion_matrices = {}
    classification_reports = {}
    optimal_thresholds = {}
    training_times = {}

    # Random Forest (Baseline)
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

    # XGBoost
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

    # LightGBM
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
            feature_cols_encoded, X_test, y_test)


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


def run_fairness_analysis(models, X_test, y_test, feature_cols_encoded):
    """Run fairness analysis on all models."""
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
                   fairness_results=None):
    """Saving model artifacts for use in dashboard."""
    output_dir = Path(os.getenv("MODELS_DIR", "models"))
    output_dir.mkdir(exist_ok=True)

    # Saving models
    joblib.dump(models["random_forest"], output_dir / "random_forest_model.pkl")
    joblib.dump(models["xgboost"], output_dir / "xgboost_model.pkl")
    joblib.dump(models["lightgbm"], output_dir / "lightgbm_model.pkl")

    # Saving encoders
    joblib.dump(salary_encoder, output_dir / "salary_encoder.pkl")
    joblib.dump(department_encoder, output_dir / "department_encoder.pkl")

    # Saving feature names
    with open(output_dir / "feature_columns.json", "w") as f:
        json.dump(feature_cols, f)

    # Saving results with training times
    for model_name in results:
        results[model_name]["training_time"] = training_times[model_name]

    with open(output_dir / "model_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Saving confusion matrices
    with open(output_dir / "confusion_matrix.json", "w") as f:
        json.dump(confusion_matrices, f, indent=2)

    # Saving classification reports
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

    # Saving optimal thresholds
    with open(output_dir / "optimal_threshold.json", "w") as f:
        json.dump(optimal_thresholds, f, indent=2)

    # Saving feature importance for all models
    for name, importance_df in feature_importances.items():
        importance_df.to_csv(output_dir / f"feature_importance_{name}.csv", index=False)

    # Saving combined comparison
    comparison_data = []
    for name, importance_df in feature_importances.items():
        top_features = importance_df.head(3)
        comparison_data.append({
            "model": name,
            "top_feature": top_features.iloc[0]["feature"],
            "top_importance": float(top_features.iloc[0]["importance"])
        })

    pd.DataFrame(comparison_data).to_csv(output_dir / "model_comparison.csv", index=False)

    # Save fairness report if available
    if fairness_results:
        with open(output_dir / "fairness_report.json", "w") as f:
            json.dump(fairness_results, f, indent=2)
        print(f"\nFairness report saved")

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
    print("=" * 70)
    print("ADVANCED EMPLOYEE ATTRITION PREDICTION MODELS")
    print("XGBoost & LightGBM vs Random Forest Baseline")
    print("=" * 70)

    # Loading data
    print("\nLoading data...")
    df, X, y, feature_cols = load_and_preprocess_data()
    print(f"Loaded {len(X)} samples with {len(feature_cols)} features")
    print(f"Attrition rate: {y.mean() * 100:.1f}%")

    # Training models
    (models, results, confusion_matrices, classification_reports,
     optimal_thresholds, training_times, salary_enc, dept_enc,
     feature_cols_encoded, X_test, y_test) = train_models(df, X, y, feature_cols)

    # Extracting feature importance
    feature_importances = extract_feature_importance(models, feature_cols_encoded)

    # Run fairness analysis
    print("\nRunning fairness analysis...")
    fairness_results = run_fairness_analysis(models, X_test, y_test, feature_cols_encoded)

    # Print fairness summary
    print("\n" + "=" * 70)
    print("FAIRNESS SUMMARY (Random Forest)")
    print("=" * 70)
    salary_ratio = fairness_results['random_forest']['by_salary']['demographic_parity_ratio']
    dept_ratio = fairness_results['random_forest']['by_department']['demographic_parity_ratio']
    print(f"\nSalary Demographic Parity Ratio: {salary_ratio:.4f}")
    print(f"  {'✓ Passes' if salary_ratio >= 0.8 else '⚠️ Fails'} EEOC 80% rule")
    print(f"\nDepartment Demographic Parity Ratio: {dept_ratio:.4f}")
    print(f"  {'✓ Passes' if dept_ratio >= 0.8 else '⚠️ Fails'} EEOC 80% rule")

    # Saving everything
    save_artifacts(models, results, feature_importances, salary_enc, dept_enc,
                   feature_cols_encoded, confusion_matrices, classification_reports,
                   optimal_thresholds, training_times, fairness_results)

    # Printing summaries
    print_summary(results, feature_importances, training_times)
    print_comparison_report(results)


if __name__ == "__main__":
    main()
