#!/usr/bin/env python3
"""
SHAP Explainability Analysis for HR Attrition Model

This script generates SHAP (SHapley Additive exPlanations) values to explain
the Random Forest model's predictions for employee attrition. It creates
visualizations and saves computed values for further analysis.

Usage:
    python scripts/explain_with_shap.py
    python scripts/explain_with_shap.py --sample-index 42  # Explain specific employee
    python scripts/explain_with_shap.py --n-samples 500    # Use fewer samples for speed
"""

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Use non-interactive backend for saving plots
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Try importing SHAP
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("ERROR: SHAP library is not installed.")
    print("Install it with: pip install shap")
    sys.exit(1)


def load_model_artifacts(models_dir: Path) -> dict:
    """
    Load trained model artifacts from the models directory.

    Args:
        models_dir: Path to the models directory

    Returns:
        Dictionary containing model, feature columns, and label encoders
    """
    print("Loading model artifacts...")

    artifacts = {}

    # Load Random Forest model
    model_path = models_dir / "random_forest_model.pkl"
    with open(model_path, 'rb') as f:
        artifacts['model'] = pickle.load(f)
    print(f"  ✓ Loaded model from {model_path}")

    # Load feature columns
    features_path = models_dir / "feature_columns.json"
    with open(features_path, 'r') as f:
        artifacts['feature_columns'] = json.load(f)
    print(f"  ✓ Loaded {len(artifacts['feature_columns'])} features")

    # Load label encoders
    le_dept_path = models_dir / "label_encoder_department.pkl"
    with open(le_dept_path, 'rb') as f:
        artifacts['le_department'] = pickle.load(f)

    le_salary_path = models_dir / "label_encoder_salary.pkl"
    with open(le_salary_path, 'rb') as f:
        artifacts['le_salary'] = pickle.load(f)
    print(f"  ✓ Loaded label encoders")

    return artifacts


def load_and_preprocess_data(data_path: Path, artifacts: dict) -> pd.DataFrame:
    """
    Load and preprocess the HR employee data.

    Args:
        data_path: Path to the CSV data file
        artifacts: Dictionary containing model artifacts (encoders, features)

    Returns:
        Preprocessed DataFrame with features only
    """
    print("Loading and preprocessing data...")

    df = pd.read_csv(data_path)
    print(f"  ✓ Loaded {len(df)} records from {data_path}")

    # Encode categorical features using the saved encoders
    if 'Department' in df.columns:
        df['Department'] = artifacts['le_department'].transform(
            df['Department'].fillna('Unknown')
        )

    if 'Salary' in df.columns:
        df['Salary'] = artifacts['le_salary'].transform(
            df['Salary'].fillna('Medium')
        )

    # Extract feature columns
    feature_cols = artifacts['feature_columns']
    X = df[feature_cols].copy()

    # Handle any missing values
    X = X.fillna(X.median())

    print(f"  ✓ Preprocessed {len(X)} samples with {len(feature_cols)} features")

    return X


def compute_shap_values(
    model,
    X: pd.DataFrame,
    n_samples: int = 1000
) -> tuple:
    """
    Compute SHAP values for the given model and data.

    Args:
        model: Trained Random Forest model
        X: Feature DataFrame
        n_samples: Number of samples to compute SHAP values for

    Returns:
        Tuple of (explainer, shap_values, sample_indices)
    """
    print(f"Computing SHAP values for {n_samples} samples...")

    # Sample data for performance
    if len(X) > n_samples:
        sample_indices = np.random.choice(len(X), n_samples, replace=False)
        X_sample = X.iloc[sample_indices]
    else:
        sample_indices = np.arange(len(X))
        X_sample = X

    # Create TreeExplainer for Random Forest
    explainer = shap.TreeExplainer(model)
    print("  ✓ Created TreeExplainer")

    # Compute SHAP values
    shap_values = explainer.shap_values(X_sample)

    # Handle binary classification output
    if isinstance(shap_values, list):
        # Use SHAP values for the positive class (attrition = 1)
        shap_values = shap_values[1]

    print(f"  ✓ Computed SHAP values with shape {shap_values.shape}")

    return explainer, shap_values, sample_indices


def create_summary_plot(
    shap_values,
    X: pd.DataFrame,
    save_path: Path
) -> None:
    """
    Create and save a SHAP summary plot (beeswarm).

    Args:
        shap_values: Computed SHAP values
        X: Feature DataFrame for the sampled data
        save_path: Path to save the plot
    """
    print(f"Creating summary plot...")

    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X, show=False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Saved summary plot to {save_path}")


def create_importance_plot(
    shap_values,
    X: pd.DataFrame,
    save_path: Path,
    top_n: int = 20
) -> None:
    """
    Create and save a SHAP feature importance bar plot.

    Args:
        shap_values: Computed SHAP values
        X: Feature DataFrame for the sampled data
        save_path: Path to save the plot
        top_n: Number of top features to display
    """
    print(f"Creating feature importance plot...")

    plt.figure(figsize=(10, 8))
    shap.plots.bar(shap_values, show=False, max_display=top_n)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Saved importance plot to {save_path}")


def create_dependence_plots(
    shap_values,
    X: pd.DataFrame,
    save_path: Path,
    top_n: int = 2
) -> None:
    """
    Create and save SHAP dependence plots for top features.

    Args:
        shap_values: Computed SHAP values
        X: Feature DataFrame for the sampled data
        save_path: Path to save the plot
        top_n: Number of top features to plot
    """
    print(f"Creating dependence plots for top {top_n} features...")

    # Calculate mean absolute SHAP values to find top features
    mean_shap = np.abs(shap_values).mean(axis=0)
    top_indices = np.argsort(mean_shap)[-top_n:][::-1]

    fig, axes = plt.subplots(1, top_n, figsize=(12, 5))

    if top_n == 1:
        axes = [axes]

    for idx, ax in enumerate(axes):
        feature_idx = top_indices[idx]
        shap.dependence_plot(
            feature_idx,
            shap_values,
            X,
            show=False,
            ax=ax
        )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Saved dependence plot to {save_path}")


def create_waterfall_plot(
    explainer,
    shap_values,
    X: pd.DataFrame,
    sample_idx: int,
    save_path: Path
) -> None:
    """
    Create and save a SHAP waterfall plot for a single prediction.

    Args:
        explainer: SHAP explainer object
        shap_values: Computed SHAP values
        X: Feature DataFrame for the sampled data
        sample_idx: Index of the sample to explain (within the sampled data)
        save_path: Path to save the plot
    """
    print(f"Creating waterfall plot for sample index {sample_idx}...")

    plt.figure(figsize=(10, 8))
    shap.plots.waterfall(
        shap_values[sample_idx],
        show=False
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Saved waterfall plot to {save_path}")


def save_shap_values(
    shap_values,
    save_path: Path
) -> None:
    """
    Save computed SHAP values as a numpy array.

    Args:
        shap_values: Computed SHAP values
        save_path: Path to save the .npy file
    """
    print(f"Saving SHAP values...")

    np.save(save_path, shap_values)
    print(f"  ✓ Saved SHAP values to {save_path}")


def save_explanation_summary(
    shap_values,
    feature_names: list,
    save_path: Path,
    top_n: int = 10
) -> None:
    """
    Save a summary of SHAP explanations as JSON.

    Args:
        shap_values: Computed SHAP values
        feature_names: List of feature names
        save_path: Path to save the JSON file
        top_n: Number of top features to include
    """
    print(f"Saving explanation summary...")

    # Calculate mean absolute SHAP values for each feature
    mean_shap = np.abs(shap_values).mean(axis=0)

    # Create feature importance ranking
    feature_importance = [
        {
            "feature": name,
            "mean_abs_shap": float(value),
            "rank": idx + 1
        }
        for idx, (name, value) in enumerate(
            sorted(
                zip(feature_names, mean_shap),
                key=lambda x: x[1],
                reverse=True
            )
        )
    ]

    summary = {
        "n_samples": len(shap_values),
        "n_features": len(feature_names),
        "top_features": feature_importance[:top_n],
        "total_shap_sum": float(np.abs(mean_shap).sum())
    }

    with open(save_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"  ✓ Saved explanation summary to {save_path}")
    print(f"\n  Top {top_n} features by importance:")
    for feat in feature_importance[:top_n]:
        print(f"    {feat['rank']}. {feat['feature']}: {feat['mean_abs_shap']:.4f}")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Generate SHAP explainability analysis for HR attrition model"
    )
    parser.add_argument(
        "--sample-index",
        type=int,
        default=None,
        help="Index of specific employee to explain (creates waterfall plot)"
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=1000,
        help="Number of samples to compute SHAP values for (default: 1000)"
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default="models",
        help="Path to models directory (default: models)"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/hr_employee_data.csv",
        help="Path to HR data CSV file (default: data/hr_employee_data.csv)"
    )

    args = parser.parse_args()

    # Setup paths
    models_dir = Path(args.models_dir)
    data_path = Path(args.data_path)

    print("=" * 60)
    print("SHAP Explainability Analysis for HR Attrition Model")
    print("=" * 60)
    print()

    # Load artifacts and data
    artifacts = load_model_artifacts(models_dir)
    X = load_and_preprocess_data(data_path, artifacts)

    print()

    # Compute SHAP values
    explainer, shap_values, sample_indices = compute_shap_values(
        artifacts['model'],
        X,
        args.n_samples
    )

    # Get sampled data for plotting
    X_sample = X.iloc[sample_indices]

    print()

    # Create visualizations
    create_summary_plot(shap_values, X_sample, models_dir / "shap_summary.png")
    create_importance_plot(shap_values, X_sample, models_dir / "shap_importance.png")
    create_dependence_plots(shap_values, X_sample, models_dir / "shap_dependence.png")

    # Save computed values
    save_shap_values(shap_values, models_dir / "shap_values.npy")
    save_explanation_summary(
        shap_values,
        artifacts['feature_columns'],
        models_dir / "shap_summary.json"
    )

    # Optional: Individual prediction explanation
    if args.sample_index is not None:
        print()
        if 0 <= args.sample_index < len(X_sample):
            create_waterfall_plot(
                explainer,
                shap_values,
                X_sample,
                args.sample_index,
                models_dir / f"shap_waterfall_{args.sample_index}.png"
            )
        else:
            print(f"  ⚠ Sample index {args.sample_index} out of range (0-{len(X_sample)-1})")

    print()
    print("=" * 60)
    print("✓ SHAP analysis complete!")
    print(f"  Outputs saved to: {models_dir}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
