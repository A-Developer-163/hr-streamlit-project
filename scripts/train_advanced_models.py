#!/usr/bin/env python3
"""
Training Advanced Employee Attrition Prediction Models
Training XGBoost and LightGBM models, comparing against Random Forest baseline.
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
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


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
    # Splitting data first to prevent data leakage
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Getting the training and test indices for proper encoding
    train_idx = X_train.index
    test_idx = X_test.index

    # Converting string columns to object dtype for LabelEncoder compatibility
    X_train["Department"] = X_train["Department"].astype(object)
    X_test["Department"] = X_test["Department"].astype(object)
    X_train["salary"] = X_train["salary"].astype(object)
    X_test["salary"] = X_test["salary"].astype(object)

    # Fitting encoders on training data only, then transforming both
    le_department = LabelEncoder()
    X_train.loc[:, "Department"] = le_department.fit_transform(X_train["Department"])
    X_test.loc[:, "Department"] = le_department.transform(X_test["Department"])

    le_salary = LabelEncoder()
    X_train.loc[:, "salary"] = le_salary.fit_transform(X_train["salary"])
    X_test.loc[:, "salary"] = le_salary.transform(X_test["salary"])

    # Encoding full X for cross-validation (same encoding as train/test)
    X_encoded = X.copy()
    X_encoded["Department"] = X_encoded["Department"].astype(object)
    X_encoded["salary"] = X_encoded["salary"].astype(object)
    X_encoded.loc[:, "Department"] = le_department.transform(X_encoded["Department"])
    X_encoded.loc[:, "salary"] = le_salary.transform(X_encoded["salary"])

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
    rf.fit(X_train, y_train)
    training_times["random_forest"] = time.time() - start_time

    rf_preds = rf.predict(X_test)
    rf_proba = rf.predict_proba(X_test)[:, 1]

    confusion_matrices["random_forest"] = confusion_matrix(y_test, rf_preds).tolist()
    classification_reports["random_forest"] = classification_report(y_test, rf_preds, output_dict=True)

    precisions, recalls, thresholds = precision_recall_curve(y_test, rf_proba)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    optimal_idx = np.argmax(f1_scores)
    optimal_thresholds["random_forest"] = float(thresholds[optimal_idx])

    results["random_forest"] = {
        "accuracy": float((rf_preds == y_test).mean()),
        "roc_auc": float(roc_auc_score(y_test, rf_proba)),
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
    xgb.fit(X_train, y_train)
    training_times["xgboost"] = time.time() - start_time

    xgb_preds = xgb.predict(X_test)
    xgb_proba = xgb.predict_proba(X_test)[:, 1]

    confusion_matrices["xgboost"] = confusion_matrix(y_test, xgb_preds).tolist()
    classification_reports["xgboost"] = classification_report(y_test, xgb_preds, output_dict=True)

    precisions, recalls, thresholds = precision_recall_curve(y_test, xgb_proba)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    optimal_idx = np.argmax(f1_scores)
    optimal_thresholds["xgboost"] = float(thresholds[optimal_idx])

    results["xgboost"] = {
        "accuracy": float((xgb_preds == y_test).mean()),
        "roc_auc": float(roc_auc_score(y_test, xgb_proba)),
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
    lgb.fit(X_train, y_train)
    training_times["lightgbm"] = time.time() - start_time

    lgb_preds = lgb.predict(X_test)
    lgb_proba = lgb.predict_proba(X_test)[:, 1]

    confusion_matrices["lightgbm"] = confusion_matrix(y_test, lgb_preds).tolist()
    classification_reports["lightgbm"] = classification_report(y_test, lgb_preds, output_dict=True)

    precisions, recalls, thresholds = precision_recall_curve(y_test, lgb_proba)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    optimal_idx = np.argmax(f1_scores)
    optimal_thresholds["lightgbm"] = float(thresholds[optimal_idx])

    results["lightgbm"] = {
        "accuracy": float((lgb_preds == y_test).mean()),
        "roc_auc": float(roc_auc_score(y_test, lgb_proba)),
        "cv_scores": cross_val_score(lgb, X_encoded, y, cv=5, scoring="roc_auc").tolist(),
        "optimal_threshold": optimal_thresholds["lightgbm"]
    }

    models["lightgbm"] = lgb

    return models, results, confusion_matrices, classification_reports, optimal_thresholds, training_times, le_department, le_salary


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


def save_artifacts(models, results, feature_importances, le_department, le_salary, feature_cols, confusion_matrices, classification_reports, optimal_thresholds, training_times):
    """Saving model artifacts for use in dashboard."""
    output_dir = Path(os.getenv("MODELS_DIR", "models"))
    output_dir.mkdir(exist_ok=True)

    # Saving models
    joblib.dump(models["random_forest"], output_dir / "random_forest_model.pkl")
    joblib.dump(models["xgboost"], output_dir / "xgboost_model.pkl")
    joblib.dump(models["lightgbm"], output_dir / "lightgbm_model.pkl")

    # Saving encoders
    joblib.dump(le_department, output_dir / "label_encoder_department.pkl")
    joblib.dump(le_salary, output_dir / "label_encoder_salary.pkl")

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
                json_reports[model_name][key] = {k: float(v) if isinstance(v, (np.integer, np.floating)) else v for k, v in value.items()}
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
    models, results, confusion_matrices, classification_reports, optimal_thresholds, training_times, le_dept, le_salary = train_models(df, X, y, feature_cols)

    # Extracting feature importance
    feature_importances = extract_feature_importance(models, feature_cols)

    # Saving everything
    save_artifacts(models, results, feature_importances, le_dept, le_salary, feature_cols, confusion_matrices, classification_reports, optimal_thresholds, training_times)

    # Printing summaries
    print_summary(results, feature_importances, training_times)
    print_comparison_report(results)

    print("\n" + "=" * 70)
    print("Optimal Thresholds (Maximize F1 Score)")
    print("=" * 70)
    for model_name, threshold in optimal_thresholds.items():
        print(f"{model_name.replace('_', ' ').title()}: {threshold:.4f}")


if __name__ == "__main__":
    main()
