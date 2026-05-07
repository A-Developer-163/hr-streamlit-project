"""Centralised configuration module for HR Employee Analytics.

Loads environment variables from .env file with backward-compatible defaults.
All paths are resolved relative to this file's location (project root).
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Get the project root (this config.py file's directory)
PROJECT_ROOT = Path(__file__).parent.resolve()

# Loading environment variables from .env file
load_dotenv()

# Data paths - resolved relative to project root
HR_DATA_PATH = str(PROJECT_ROOT / os.getenv("HR_DATA_PATH", "data/hr_employee_data.csv"))
HR_EXCEL_DATA_PATH = str(PROJECT_ROOT / os.getenv("HR_EXCEL_DATA_PATH", "data/hr_employee_data.xlsx"))
DATA_DIR = PROJECT_ROOT / os.getenv("DATA_DIR", "data")

# Model artifacts
MODELS_DIR = PROJECT_ROOT / os.getenv("MODELS_DIR", "models")

# Output paths
DRIFT_PLOT_PATH = str(PROJECT_ROOT / os.getenv("DRIFT_PLOT_PATH", "models/drift_plot.png"))
SHAP_OUTPUT_DIR = PROJECT_ROOT / os.getenv("SHAP_OUTPUT_DIR", "models/shap")
