#!/usr/bin/env python3
"""
Data exploration script for HR Employee Analytics.
Analyzes the dataset to discover potential data science project opportunities.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from config import HR_DATA_PATH


def analyze_dataset(csv_path: str = None) -> None:
    """Comprehensive analysis of HR employee dataset.

    Args:
        csv_path: Path to CSV file (defaults to config.HR_DATA_PATH)
    """
    if csv_path is None:
        csv_path = HR_DATA_PATH
    """Comprehensive analysis of HR employee dataset."""

    print("=" * 60)
    print("HR EMPLOYEE DATA EXPLORATION")
    print("=" * 60)

    # Load data
    df = pd.read_csv(csv_path)

    # 1. Basic Info
    print("\n[1] DATASET OVERVIEW")
    print("-" * 40)
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024:.1f} KB")

    # 2. Column Analysis
    print("\n[2] COLUMNS & DATA TYPES")
    print("-" * 40)
    for col in df.columns:
        dtype = str(df[col].dtype)
        unique_count = df[col].nunique()
        null_count = df[col].isnull().sum()
        null_pct = (null_count / len(df)) * 100
        print(f"  {col:30s} | {dtype:10s} | Unique: {unique_count:5d} | Null: {null_count:5d} ({null_pct:5.1f}%)")

    # 3. Identify Column Types
    print("\n[3] COLUMN CLASSIFICATION")
    print("-" * 40)

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category', 'string']).columns.tolist()
    date_cols = df.select_dtypes(include=['datetime64']).columns.tolist()

    print(f"  Numeric columns:     {len(numeric_cols)}")
    for col in numeric_cols[:10]:
        print(f"    - {col}")
    if len(numeric_cols) > 10:
        print(f"    ... and {len(numeric_cols) - 10} more")

    print(f"\n  Categorical columns: {len(categorical_cols)}")
    for col in categorical_cols[:10]:
        unique_vals = df[col].unique()[:5]
        print(f"    - {col} (examples: {list(unique_vals)})")
    if len(categorical_cols) > 10:
        print(f"    ... and {len(categorical_cols) - 10} more")

    if date_cols:
        print(f"\n  Date columns:        {len(date_cols)}")
        for col in date_cols:
            print(f"    - {col}")

    # 4. Numeric Statistics
    print("\n[4] NUMERIC STATISTICS")
    print("-" * 40)
    stats = df[numeric_cols].describe().T
    for col in numeric_cols[:15]:
        if col in stats.index:
            row = stats.loc[col]
            print(f"  {col:30s}")
            print(f"    Min: {row['min']:>12.2f} | Max: {row['max']:>12.2f} | Mean: {row['mean']:>10.2f}")

    # 5. Categorical Value Counts
    print("\n[5] CATEGORICAL DISTRIBUTIONS")
    print("-" * 40)
    for col in categorical_cols[:8]:
        value_counts = df[col].value_counts()
        print(f"\n  {col}:")
        for val, count in value_counts.head(5).items():
            pct = (count / len(df)) * 100
            print(f"    {val:30s}: {count:5d} ({pct:5.1f}%)")

    # 6. Data Quality Issues
    print("\n[6] DATA QUALITY")
    print("-" * 40)
    null_counts = df.isnull().sum()
    cols_with_nulls = null_counts[null_counts > 0]
    if len(cols_with_nulls) > 0:
        print("  Columns with missing values:")
        for col, count in cols_with_nulls.items():
            pct = (count / len(df)) * 100
            print(f"    {col:30s}: {count:5d} ({pct:5.1f}%)")
    else:
        print("  ✓ No missing values")

    # Check for duplicates
    dup_count = df.duplicated().sum()
    if dup_count > 0:
        print(f"  ⚠ Found {dup_count} duplicate rows")
    else:
        print("  ✓ No duplicate rows")

    # 7. Project Opportunities
    print("\n[7] POTENTIAL DATA SCIENCE PROJECTS")
    print("-" * 40)

    projects = []
    col_names_lower = [c.lower() for c in df.columns]

    # Look for target variables
    if 'attrition' in df.columns or 'left' in col_names_lower or 'turnover' in col_names_lower:
        projects.append("Employee Attrition Prediction (Classification)")
        target_col = 'attrition' if 'attrition' in df.columns else ('left' if 'left' in df.columns else None)
        if target_col and df[target_col].nunique() == 2:
            attrition_rate = df[target_col].mean() * 100
            print(f"  [*] Target variable found: '{target_col}'")
            print(f"  [*] Attrition rate: {attrition_rate:.1f}%")

    if 'salary' in df.columns or 'income' in col_names_lower:
        projects.append("Salary Analysis & Compensation Benchmarking")

    if 'performance' in col_names_lower or 'rating' in col_names_lower or 'last_evaluation' in df.columns:
        projects.append("Employee Performance Analysis")

    if 'department' in df.columns or 'dept' in df.columns:
        projects.append("Department-wise HR Analytics Dashboard")

    if 'satisfaction' in df.columns or 'satisfaction_level' in df.columns:
        projects.append("Employee Satisfaction Analysis")

    # Always applicable
    projects.extend([
        "Exploratory Data Analysis Dashboard",
        "Employee Demographics & Segmentation",
        "Tenure & Retention Analysis",
    ])

    print()
    for i, project in enumerate(projects, 1):
        print(f"  {i}. {project}")

    # 8. Suggested Visualizations
    print("\n[8] SUGGESTED VISUALIZATIONS")
    print("-" * 40)
    print("  Distribution plots for:")
    for col in numeric_cols[:5]:
        print(f"    - {col}")

    print("\n  Bar charts for:")
    for col in categorical_cols[:5]:
        print(f"    - {col}")

    if len(numeric_cols) >= 2:
        print("\n  Correlation analysis between numeric features")
        print("  Scatter plots for relationships")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    analyze_dataset()
