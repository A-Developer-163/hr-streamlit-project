#!/usr/bin/env python3
"""
Train Employee Attrition Prediction Model
Creates and evaluates classification model, saves model and insights.
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import json

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve


def load_and_preprocess_data(csv_path: str = "data/hr_employee_data.csv"):
    """Load and preprocess data for modeling."""
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
    # Split data first to prevent data leakage
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Get the training and test indices for proper encoding
    train_idx = X_train.index
    test_idx = X_test.index

    # Convert string columns to object dtype for LabelEncoder compatibility
    X_train["Department"] = X_train["Department"].astype(object)
    X_test["Department"] = X_test["Department"].astype(object)
    X_train["salary"] = X_train["salary"].astype(object)
    X_test["salary"] = X_test["salary"].astype(object)

    # Fit encoders on training data only, then transform both
    le_department = LabelEncoder()
    X_train.loc[:, "Department"] = le_department.fit_transform(X_train["Department"])
    X_test.loc[:, "Department"] = le_department.transform(X_test["Department"])

    le_salary = LabelEncoder()
    X_train.loc[:, "salary"] = le_salary.fit_transform(X_train["salary"])
    X_test.loc[:, "salary"] = le_salary.transform(X_test["salary"])

    # Encode full X for cross-validation (same encoding as train/test)
    X_encoded = X.copy()
    X_encoded["Department"] = X_encoded["Department"].astype(object)
    X_encoded["salary"] = X_encoded["salary"].astype(object)
    X_encoded.loc[:, "Department"] = le_department.transform(X_encoded["Department"])
    X_encoded.loc[:, "salary"] = le_salary.transform(X_encoded["salary"])

    # Scale features for Logistic Regression
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

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
    rf.fit(X_train, y_train)
    rf_preds = rf.predict(X_test)
    rf_proba = rf.predict_proba(X_test)[:, 1]

    # Confusion Matrix
    cm_rf = confusion_matrix(y_test, rf_preds)
    confusion_matrices["random_forest"] = cm_rf.tolist()

    # Classification Report
    cr_rf = classification_report(y_test, rf_preds, output_dict=True)
    classification_reports["random_forest"] = cr_rf

    # Find optimal threshold using precision-recall curve
    precisions, recalls, thresholds = precision_recall_curve(y_test, rf_proba)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    optimal_idx = np.argmax(f1_scores)
    optimal_thresholds["random_forest"] = float(thresholds[optimal_idx])

    models["random_forest"] = {
        "model": rf,
        "scaler": None,
        "predictions": rf_preds,
        "probabilities": rf_proba
    }

    results["random_forest"] = {
        "accuracy": (rf_preds == y_test).mean(),
        "roc_auc": roc_auc_score(y_test, rf_proba),
        "cv_scores": cross_val_score(rf, X_encoded, y, cv=5, scoring="roc_auc").tolist(),
        "optimal_threshold": optimal_thresholds["random_forest"]
    }

    # Logistic Regression
    print("Training Logistic Regression...")
    lr = LogisticRegression(
        random_state=42,
        max_iter=1000,
        class_weight='balanced'
    )
    lr.fit(X_train_scaled, y_train)
    lr_preds = lr.predict(X_test_scaled)
    lr_proba = lr.predict_proba(X_test_scaled)[:, 1]

    # Confusion Matrix
    cm_lr = confusion_matrix(y_test, lr_preds)
    confusion_matrices["logistic_regression"] = cm_lr.tolist()

    # Classification Report
    cr_lr = classification_report(y_test, lr_preds, output_dict=True)
    classification_reports["logistic_regression"] = cr_lr

    # Find optimal threshold using precision-recall curve
    precisions, recalls, thresholds = precision_recall_curve(y_test, lr_proba)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    optimal_idx = np.argmax(f1_scores)
    optimal_thresholds["logistic_regression"] = float(thresholds[optimal_idx])

    models["logistic_regression"] = {
        "model": lr,
        "scaler": scaler,
        "predictions": lr_preds,
        "probabilities": lr_proba
    }

    # CV for LR requires scaling - use pipeline
    lr_pipeline = make_pipeline(StandardScaler(), LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced'))
    results["logistic_regression"] = {
        "accuracy": (lr_preds == y_test).mean(),
        "roc_auc": roc_auc_score(y_test, lr_proba),
        "cv_scores": cross_val_score(lr_pipeline, X_encoded, y, cv=5, scoring="roc_auc").tolist(),
        "optimal_threshold": optimal_thresholds["logistic_regression"]
    }

    return models, results, X_test, y_test, confusion_matrices, classification_reports, optimal_thresholds, le_department, le_salary


def extract_feature_importance(model, feature_cols):
    """Extract and format feature importance."""
    importances = pd.DataFrame({
        "feature": feature_cols,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)

    return importances


def save_artifacts(models, results, feature_importance, le_department, le_salary, feature_cols, confusion_matrices, classification_reports, optimal_thresholds):
    """Save model artifacts for use in dashboard."""
    output_dir = Path("models")
    output_dir.mkdir(exist_ok=True)

    # Save models
    joblib.dump(models["random_forest"]["model"], output_dir / "random_forest_model.pkl")
    joblib.dump(models["logistic_regression"]["model"], output_dir / "logistic_regression_model.pkl")
    joblib.dump(models["logistic_regression"]["scaler"], output_dir / "scaler.pkl")

    # Save encoders
    joblib.dump(le_department, output_dir / "label_encoder_department.pkl")
    joblib.dump(le_salary, output_dir / "label_encoder_salary.pkl")

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
                json_reports[model_name][key] = {k: float(v) if isinstance(v, (np.integer, np.floating)) else v for k, v in value.items()}
            else:
                json_reports[model_name][key] = float(value) if isinstance(value, (np.integer, np.floating)) else value

    with open(output_dir / "classification_report.json", "w") as f:
        json.dump(json_reports, f, indent=2)

    # Save optimal thresholds
    with open(output_dir / "optimal_threshold.json", "w") as f:
        json.dump(optimal_thresholds, f, indent=2)

    # Save feature importance
    feature_importance.to_csv(output_dir / "feature_importance.csv", index=False)

    print(f"\nModel artifacts saved to {output_dir}/")


def print_summary(results, feature_importance):
    """Print model performance summary."""
    print("\n" + "=" * 60)
    print("MODEL PERFORMANCE SUMMARY")
    print("=" * 60)

    for model_name, metrics in results.items():
        print(f"\n{model_name.replace('_', ' ').title()}:")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  ROC AUC:   {metrics['roc_auc']:.4f}")
        print(f"  CV Scores: {np.array(metrics['cv_scores']).mean():.4f} (+/- {np.array(metrics['cv_scores']).std() * 2:.4f})")

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
    models, results, X_test, y_test, confusion_matrices, classification_reports, optimal_thresholds, le_dept, le_salary = train_models(df, X, y)

    # Extract feature importance from Random Forest
    feature_importance = extract_feature_importance(
        models["random_forest"]["model"],
        feature_cols
    )

    # Save everything
    save_artifacts(models, results, feature_importance, le_dept, le_salary, feature_cols, confusion_matrices, classification_reports, optimal_thresholds)

    # Print summary
    print_summary(results, feature_importance)

    # Print optimal thresholds
    print("\n" + "=" * 60)
    print("OPTIMAL THRESHOLDS (Maximize F1 Score)")
    print("=" * 60)
    for model_name, threshold in optimal_thresholds.items():
        print(f"{model_name.replace('_', ' ').title()}: {threshold:.4f}")


if __name__ == "__main__":
    main()
