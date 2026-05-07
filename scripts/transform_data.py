#!/usr/bin/env python3
"""
Transform HR Employee data from Excel to CSV format.
Run this script to convert the source Excel file to optimized CSV.
"""

import pandas as pd
import os
from pathlib import Path
from config import HR_EXCEL_DATA_PATH, HR_DATA_PATH


def convert_excel_to_csv(
    excel_path: str = None,
    csv_path: str = None,
) -> None:
    """Convert Excel file to CSV with optimizations.

    Args:
        excel_path: Path to Excel file (defaults to config.HR_EXCEL_DATA_PATH)
        csv_path: Path to output CSV file (defaults to config.HR_DATA_PATH)
    """
    # Use config defaults if not provided
    if excel_path is None:
        excel_path = HR_EXCEL_DATA_PATH
    if csv_path is None:
        csv_path = HR_DATA_PATH
    """Convert Excel file to CSV with optimizations."""
    print(f"Loading Excel file: {excel_path}")

    if not os.path.exists(excel_path):
        raise FileNotFoundError(f"Excel file not found: {excel_path}")

    # Load Excel file
    df = pd.read_excel(excel_path, engine="openpyxl")

    # # Changing average_montly_average_montly_hours to average_monthly_hours
    # df.rename(columns={"average_montly_hours": "average_monthly_hours"})

    # # Standardising the column names
    # df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')

    # print(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    # for col in df.columns:
    #     print(col)

    # Optimize data types
    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = df[col].astype("category")

    # Ensure output directory exists
    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)

    # Save to CSV
    df.to_csv(csv_path, index=False)

    print(f"Saved to: {csv_path}")
    print(f"File size: {os.path.getsize(csv_path) / 1024:.1f} KB")


if __name__ == "__main__":
    convert_excel_to_csv()
