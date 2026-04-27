"""
Test data loading functionality
"""

import pandas as pd
import pytest


def test_hr_data_file_exists():
    """Test that the HR data file exists."""
    import os
    assert os.path.exists("data/hr_employee_data.xlsx"), "HR data file not found"


def test_load_hr_data():
    """Test loading the HR Excel file."""
    df = pd.read_excel("data/hr_employee_data.xlsx")
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    assert len(df.columns) > 0


def test_data_has_employee_id(sample_employee_data):
    """Test that sample data has EmployeeID column."""
    assert "EmployeeID" in sample_employee_data.columns


def test_data_not_empty(sample_employee_data):
    """Test that sample data is not empty."""
    assert len(sample_employee_data) > 0
