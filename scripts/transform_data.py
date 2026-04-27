#!/usr/bin/env python3
"""
Transform HR Employee data from Excel to CSV format.
Run this script to convert the source Excel file to optimized CSV.
"""

import pandas as pd
import os
from pathlib import Path


def convert_excel_to_csv(
    excel_path: str = "data/hr_employee_data.xlsx",
    csv_path: str = "data/hr_employee_data.csv",
) -> None:
    """Convert Excel file to CSV with optimizations."""
    print(f"Loading Excel file: {excel_path}")

    if not os.path.exists(excel_path):
        raise FileNotFoundError(f"Excel file not found: {excel_path}")

    # Load Excel file
    df = pd.read_excel(excel_path, engine="openpyxl")

    print(f"Loaded {len(df)} rows, {len(df.columns)} columns")

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
