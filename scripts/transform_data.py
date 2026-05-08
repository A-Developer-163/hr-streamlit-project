#!/usr/bin/env python3
"""
Transform HR Employee data from Excel to CSV format.
Standardises column names to follow Pythonic snake_case conventions
with correct spelling and consistent naming patterns.
"""

import pandas as pd
import os
from pathlib import Path
from typing import Dict
from config import HR_EXCEL_DATA_PATH, HR_DATA_PATH


# Column mapping: old_name -> new_name
COLUMN_MAPPING: Dict[str, str] = {
    "Emp_Id": "employee_id",
    "number_project": "num_projects",
    "average_montly_hours": "avg_monthly_hours",
    "time_spend_company": "years_at_company",
    "Work_accident": "had_work_accident",
    "left": "attrition",
    "promotion_last_5years": "promotion_last_5_years",
    "Department": "department",
    # Unchanged columns (explicitly mapped for clarity)
    "satisfaction_level": "satisfaction_level",
    "last_evaluation": "last_evaluation",
    "salary": "salary"
}


def convert_excel_to_csv(
    excel_path: str = None,
    csv_path: str = None,
) -> None:
    """Convert Excel file to CSV with standardised column names.

    Args:
        excel_path: Path to Excel file (defaults to config.HR_EXCEL_DATA_PATH)
        csv_path: Path to output CSV file (defaults to config.HR_DATA_PATH)
    """
    # Use config defaults if not provided
    if excel_path is None:
        excel_path = HR_EXCEL_DATA_PATH
    if csv_path is None:
        csv_path = HR_DATA_PATH

    print(f"Loading Excel file: {excel_path}")

    if not os.path.exists(excel_path):
        raise FileNotFoundError(f"Excel file not found: {excel_path}")

    # Load Excel file
    df = pd.read_excel(excel_path, engine="openpyxl")

    original_row_count = len(df)
    original_columns = list(df.columns)

    # Renaming columns to new Pythonic names
    df.rename(columns=COLUMN_MAPPING, inplace=True)

    print(f"\nColumn renaming complete:")
    for old, new in COLUMN_MAPPING.items():
        if old != new:
            print(f"  {old:30} -> {new}")

    # Optimize data types
    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = df[col].astype("category")

    # Ensure output directory exists
    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)

    # Save to CSV
    df.to_csv(csv_path, index=False)

    print(f"\nSaved to: {csv_path}")
    print(f"Rows: {original_row_count}")
    print(f"Columns: {len(original_columns)} -> {len(df.columns)}")
    print(f"File size: {os.path.getsize(csv_path) / 1024:.1f} KB")


if __name__ == "__main__":
    convert_excel_to_csv()
