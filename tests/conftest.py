"""
Test fixtures for HR Employee Analytics
"""

import pytest
import pandas as pd


@pytest.fixture
def hr_data_columns():
    """Expected columns from HR Employee Descriptive Analytics."""
    return [
        "EmployeeID",
        "Name",
        "Department",
        "Age",
        "Gender",
        "Salary",
        "JoinDate",
        "YearsOfService",
    ]


@pytest.fixture
def sample_employee_data():
    """Create minimal sample data for testing."""
    return pd.DataFrame({
        "EmployeeID": [1, 2, 3],
        "Name": ["Alice", "Bob", "Charlie"],
        "Department": ["HR", "Engineering", "Sales"],
        "Age": [30, 35, 28],
        "Gender": ["F", "M", "M"],
        "Salary": [60000, 80000, 55000],
    })
