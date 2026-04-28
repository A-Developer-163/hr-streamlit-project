#!/usr/bin/env python3
"""
Data Drift Detection Script for HR Employee Attrition Model

This script detects data drift by comparing new data against baseline training data.
It computes statistical drift metrics for both numeric and categorical features and
generates a comprehensive drift report with visualizations.

Usage:
    python scripts/detect_drift.py --new-data data/new_hr_data.csv
    python scripts/detect_drift.py  # Uses training data (should show no drift)
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


def compute_psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """
    Compute Population Stability Index (PSI) for numeric features.

    PSI measures how much a variable has shifted in distribution.
    Values > 0.2 indicate significant drift.

    Args:
        expected: Baseline distribution values
        actual: New distribution values
        bins: Number of bins for discretization

    Returns:
        PSI value
    """
    # Create bins based on expected data
    min_val = min(expected.min(), actual.min())
    max_val = max(expected.max(), actual.max())

    # Handle edge case where all values are the same
    if min_val == max_val:
        return 0.0

    bins_edges = np.linspace(min_val, max_val, bins + 1)

    # Calculate histograms
    expected_counts, _ = np.histogram(expected, bins=bins_edges)
    actual_counts, _ = np.histogram(actual, bins=bins_edges)

    # Avoid division by zero
    expected_percents = expected_counts / len(expected)
    actual_percents = actual_counts / len(actual)

    # Replace zeros with small value to avoid log(0)
    expected_percents = np.where(expected_percents == 0, 0.0001, expected_percents)
    actual_percents = np.where(actual_percents == 0, 0.0001, actual_percents)

    # Compute PSI
    psi = np.sum((actual_percents - expected_percents) * np.log(actual_percents / expected_percents))

    return psi


def compute_js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """
    Compute Jensen-Shannon divergence for categorical features.

    JS divergence measures the similarity between two probability distributions.
    Values > 0.1 indicate moderate drift.

    Args:
        p: Baseline distribution probabilities
        q: New distribution probabilities

    Returns:
        JS divergence value
    """
    # Ensure both arrays have the same categories
    all_keys = sorted(set(p.keys()).union(set(q.keys())))

    p_vals = np.array([p.get(k, 0) for k in all_keys], dtype=float)
    q_vals = np.array([q.get(k, 0) for k in all_keys], dtype=float)

    # Normalize to ensure sum = 1
    p_vals = p_vals / p_vals.sum()
    q_vals = q_vals / q_vals.sum()

    # Compute KL divergences
    def kl_div(a, b):
        a = np.where(a == 0, 1e-10, a)
        b = np.where(b == 0, 1e-10, b)
        return np.sum(a * np.log(a / b))

    m = (p_vals + q_vals) / 2
    js = 0.5 * kl_div(p_vals, m) + 0.5 * kl_div(q_vals, m)

    return js


def compute_baseline_stats(df: pd.DataFrame, exclude_cols: List[str] = None) -> Dict[str, Any]:
    """
    Compute baseline statistics for all features.

    Args:
        df: Input dataframe
        exclude_cols: Columns to exclude from analysis

    Returns:
        Dictionary containing baseline statistics
    """
    if exclude_cols is None:
        exclude_cols = ['Emp_Id']

    baseline = {}

    # Identify numeric and categorical columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

    # Remove excluded columns
    numeric_cols = [col for col in numeric_cols if col not in exclude_cols]
    categorical_cols = [col for col in categorical_cols if col not in exclude_cols]

    # Compute statistics for numeric features
    baseline['numeric'] = {}
    for col in numeric_cols:
        baseline['numeric'][col] = {
            'mean': float(df[col].mean()),
            'std': float(df[col].std()),
            'min': float(df[col].min()),
            'max': float(df[col].max()),
            'q25': float(df[col].quantile(0.25)),
            'q50': float(df[col].quantile(0.50)),
            'q75': float(df[col].quantile(0.75)),
            'values': df[col].tolist()  # Store all values for PSI computation
        }

    # Compute statistics for categorical features
    baseline['categorical'] = {}
    for col in categorical_cols:
        value_counts = df[col].value_counts()
        baseline['categorical'][col] = {
            'frequencies': {str(k): int(v) for k, v in value_counts.items()},
            'unique_count': int(df[col].nunique())
        }

    return baseline


def detect_drift(baseline: Dict[str, Any], new_df: pd.DataFrame,
                 exclude_cols: List[str] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Detect data drift by comparing new data against baseline.

    Args:
        baseline: Baseline statistics dictionary
        new_df: New data dataframe
        exclude_cols: Columns to exclude from analysis

    Returns:
        Tuple of (drift summary dataframe, detailed report dictionary)
    """
    if exclude_cols is None:
        exclude_cols = ['Emp_Id']

    drift_results = []
    detailed_report = {
        'drift_detected': False,
        'features': {}
    }

    # Process numeric features
    for col, stats in baseline.get('numeric', {}).items():
        if col not in new_df.columns:
            drift_results.append({
                'feature': col,
                'type': 'numeric',
                'metric': 'PSI',
                'value': np.nan,
                'status': 'Missing',
                'threshold': '< 0.2'
            })
            detailed_report['features'][col] = {
                'status': 'Missing',
                'error': f'Column not found in new data'
            }
            continue

        new_values = new_df[col].dropna().values
        baseline_values = np.array(stats['values'])

        psi = compute_psi(baseline_values, new_values)

        # Determine status based on PSI
        if psi > 0.2:
            status = 'Drift'
            detailed_report['drift_detected'] = True
        elif psi > 0.1:
            status = 'Warning'
        else:
            status = 'OK'

        drift_results.append({
            'feature': col,
            'type': 'numeric',
            'metric': 'PSI',
            'value': round(psi, 4),
            'status': status,
            'threshold': '< 0.2'
        })

        detailed_report['features'][col] = {
            'status': status,
            'psi': psi,
            'baseline_mean': stats['mean'],
            'new_mean': float(new_values.mean()),
            'baseline_std': stats['std'],
            'new_std': float(new_values.std())
        }

    # Process categorical features
    for col, stats in baseline.get('categorical', {}).items():
        if col not in new_df.columns:
            drift_results.append({
                'feature': col,
                'type': 'categorical',
                'metric': 'JS Divergence',
                'value': np.nan,
                'status': 'Missing',
                'threshold': '< 0.1'
            })
            detailed_report['features'][col] = {
                'status': 'Missing',
                'error': f'Column not found in new data'
            }
            continue

        baseline_freqs = stats['frequencies']
        new_freqs = new_df[col].value_counts().to_dict()
        new_freqs = {str(k): int(v) for k, v in new_freqs.items()}

        js_div = compute_js_divergence(baseline_freqs, new_freqs)

        # Determine status based on JS divergence
        if js_div > 0.1:
            status = 'Drift'
            detailed_report['drift_detected'] = True
        elif js_div > 0.05:
            status = 'Warning'
        else:
            status = 'OK'

        # Check for new categories
        new_categories = set(new_freqs.keys()) - set(baseline_freqs.keys())
        if new_categories:
            status = 'Warning'
            detailed_report['features'][col] = {
                'status': status,
                'js_divergence': js_div,
                'new_categories': list(new_categories)
            }
        else:
            detailed_report['features'][col] = {
                'status': status,
                'js_divergence': js_div,
                'baseline_top_categories': dict(list(sorted(
                    baseline_freqs.items(), key=lambda x: x[1], reverse=True
                ))[:5]),
                'new_top_categories': dict(list(sorted(
                    new_freqs.items(), key=lambda x: x[1], reverse=True
                ))[:5])
            }

        drift_results.append({
            'feature': col,
            'type': 'categorical',
            'metric': 'JS Divergence',
            'value': round(js_div, 4),
            'status': status,
            'threshold': '< 0.1'
        })

    drift_df = pd.DataFrame(drift_results)
    return drift_df, detailed_report


def create_drift_visualization(baseline: Dict[str, Any], new_df: pd.DataFrame,
                               output_path: str = 'models/drift_plot.png'):
    """
    Create visualization comparing baseline and new data distributions.

    Args:
        baseline: Baseline statistics dictionary
        new_df: New data dataframe
        output_path: Path to save the plot
    """
    # Determine number of subplots needed
    numeric_cols = list(baseline.get('numeric', {}).keys())
    categorical_cols = list(baseline.get('categorical', {}).keys())

    if not numeric_cols and not categorical_cols:
        print("No features to visualize.")
        return

    # Create figure with subplots
    n_numeric = min(len(numeric_cols), 6)  # Limit to 6 numeric plots
    n_categorical = min(len(categorical_cols), 4)  # Limit to 4 categorical plots

    if n_numeric > 0:
        fig, axes = plt.subplots(n_numeric, 1, figsize=(12, 4 * n_numeric))
        if n_numeric == 1:
            axes = [axes]
    else:
        fig, axes = plt.subplots(n_categorical, 1, figsize=(12, 4 * n_categorical))
        if n_categorical == 1:
            axes = [axes]

    fig.suptitle('Data Drift Visualization: Baseline vs New Data', fontsize=16, y=0.995)

    plot_idx = 0

    # Plot numeric features
    for col in numeric_cols[:n_numeric]:
        ax = axes[plot_idx]
        baseline_values = np.array(baseline['numeric'][col]['values'])
        new_values = new_df[col].dropna().values

        ax.hist(baseline_values, bins=30, alpha=0.5, label='Baseline', color='blue')
        ax.hist(new_values, bins=30, alpha=0.5, label='New Data', color='orange')
        ax.set_title(f'{col} Distribution')
        ax.set_xlabel(col)
        ax.set_ylabel('Frequency')
        ax.legend()
        plot_idx += 1

    # Plot categorical features if we have space
    if n_categorical > 0 and plot_idx < len(axes):
        for col in categorical_cols[:n_categorical]:
            if plot_idx >= len(axes):
                break
            ax = axes[plot_idx]

            baseline_freqs = baseline['categorical'][col]['frequencies']
            new_freqs = new_df[col].value_counts().to_dict()

            # Get top categories
            all_categories = sorted(
                set(baseline_freqs.keys()).union(set(new_freqs.keys())),
                key=lambda x: baseline_freqs.get(x, 0) + new_freqs.get(x, 0),
                reverse=True
            )[:10]

            x = np.arange(len(all_categories))
            width = 0.35

            baseline_vals = [baseline_freqs.get(cat, 0) for cat in all_categories]
            new_vals = [new_freqs.get(cat, 0) for cat in all_categories]

            ax.bar(x - width/2, baseline_vals, width, label='Baseline', alpha=0.8)
            ax.bar(x + width/2, new_vals, width, label='New Data', alpha=0.8)
            ax.set_title(f'{col} Distribution (Top 10)')
            ax.set_xlabel(col)
            ax.set_ylabel('Frequency')
            ax.set_xticks(x)
            ax.set_xticklabels(all_categories, rotation=45, ha='right')
            ax.legend()
            plot_idx += 1

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nDrift visualization saved to: {output_path}")
    plt.close()


def print_drift_summary(drift_df: pd.DataFrame):
    """
    Print a human-readable summary of drift detection results.

    Args:
        drift_df: Drift summary dataframe
    """
    print("\n" + "="*80)
    print("DATA DRIFT DETECTION REPORT")
    print("="*80)

    # Count statuses
    n_drift = (drift_df['status'] == 'Drift').sum()
    n_warning = (drift_df['status'] == 'Warning').sum()
    n_ok = (drift_df['status'] == 'OK').sum()
    n_missing = (drift_df['status'] == 'Missing').sum()

    print(f"\nSummary: {n_ok} OK, {n_warning} Warnings, {n_drift} Drift Detected, {n_missing} Missing")

    if n_drift > 0:
        print("\n⚠️  SIGNIFICANT DRIFT DETECTED!")
        drift_features = drift_df[drift_df['status'] == 'Drift']['feature'].tolist()
        print(f"Features with drift: {', '.join(drift_features)}")

    if n_warning > 0:
        print("\n⚠️  Warnings:")
        warning_features = drift_df[drift_df['status'] == 'Warning']['feature'].tolist()
        print(f"Features with warnings: {', '.join(warning_features)}")

    print("\n" + "-"*80)
    print(f"{'Feature':<25} {'Type':<12} {'Metric':<15} {'Value':<10} {'Status':<10}")
    print("-"*80)

    for _, row in drift_df.iterrows():
        value_str = f"{row['value']:.4f}" if not pd.isna(row['value']) else "N/A"
        status_symbol = "✓" if row['status'] == "OK" else "⚠" if row['status'] == "Warning" else "✗" if row['status'] == "Drift" else "?"
        print(f"{row['feature']:<25} {row['type']:<12} {row['metric']:<15} {value_str:<10} {status_symbol} {row['status']}")

    print("-"*80)
    print(f"\nThresholds:")
    print(f"  - Numeric (PSI): < 0.1 OK, 0.1-0.2 Warning, > 0.2 Drift")
    print(f"  - Categorical (JS): < 0.05 OK, 0.05-0.1 Warning, > 0.1 Drift")
    print("="*80 + "\n")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Detect data drift in HR employee data compared to baseline.'
    )
    parser.add_argument(
        '--new-data',
        type=str,
        default=None,
        help='Path to new data CSV file. If not provided, uses training data.'
    )
    parser.add_argument(
        '--baseline-data',
        type=str,
        default='data/hr_employee_data.csv',
        help='Path to baseline/training data CSV file (default: data/hr_employee_data.csv)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='models',
        help='Directory to save output files (default: models)'
    )
    parser.add_argument(
        '--skip-visualization',
        action='store_true',
        help='Skip generating drift visualization plot'
    )

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load baseline data
    print(f"Loading baseline data from: {args.baseline_data}")
    baseline_df = pd.read_csv(args.baseline_data)
    print(f"Baseline data shape: {baseline_df.shape}")

    # Compute and save baseline statistics
    print("\nComputing baseline statistics...")
    baseline_stats = compute_baseline_stats(baseline_df)

    baseline_path = output_dir / 'baseline_stats.json'
    # Exclude raw values from saved baseline to reduce file size
    baseline_to_save = {
        'numeric': {
            col: {k: v for k, v in stats.items() if k != 'values'}
            for col, stats in baseline_stats['numeric'].items()
        },
        'categorical': baseline_stats['categorical']
    }
    with open(baseline_path, 'w') as f:
        json.dump(baseline_to_save, f, indent=2)
    print(f"Baseline statistics saved to: {baseline_path}")

    # Load or use new data
    if args.new_data:
        print(f"\nLoading new data from: {args.new_data}")
        new_df = pd.read_csv(args.new_data)
        print(f"New data shape: {new_df.shape}")
    else:
        print("\nNo new data provided. Using baseline data (should show no drift).")
        new_df = baseline_df.copy()

    # Detect drift
    print("\nDetecting drift...")
    drift_df, detailed_report = detect_drift(baseline_stats, new_df)

    # Save detailed report
    report_path = output_dir / 'drift_report.json'
    with open(report_path, 'w') as f:
        json.dump(detailed_report, f, indent=2)
    print(f"Detailed drift report saved to: {report_path}")

    # Create visualization
    if not args.skip_visualization:
        print("\nGenerating drift visualization...")
        plot_path = output_dir / 'drift_plot.png'
        create_drift_visualization(baseline_stats, new_df, str(plot_path))

    # Print summary
    print_drift_summary(drift_df)

    # Exit with code 1 if drift detected
    if detailed_report['drift_detected']:
        print("⚠️  Data drift detected! Please review the model performance.")
        sys.exit(1)
    else:
        print("✓ No significant data drift detected.")
        sys.exit(0)


if __name__ == '__main__':
    main()
