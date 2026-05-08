"""Centralised configuration module for HR Employee Analytics.

Loads environment variables from .env file with backward-compatible defaults.
All paths are resolved relative to this file's location (project root).
"""
import os
import joblib
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


def save_preprocessing_artifacts(scaler, salary_encoder, department_encoder, feature_cols, output_dir=None):
    """Save all preprocessing artifacts to a single file for faster loading.

    Args:
        scaler: StandardScaler for logistic regression features
        salary_encoder: OrdinalEncoder for salary levels
        department_encoder: OneHotEncoder for department categories
        feature_cols: List of encoded feature column names
        output_dir: Directory to save artifacts (defaults to MODELS_DIR)
    """
    if output_dir is None:
        output_dir = MODELS_DIR
    output_dir.mkdir(exist_ok=True)

    artifacts = {
        "scaler": scaler,
        "salary_encoder": salary_encoder,
        "department_encoder": department_encoder,
        "feature_cols": feature_cols
    }
    joblib.dump(artifacts, output_dir / "preprocessing_artifacts.pkl")


def load_preprocessing_artifacts_combined(models_dir=None):
    """Load all preprocessing artifacts from a single file.

    Args:
        models_dir: Directory containing artifacts (defaults to MODELS_DIR)

    Returns:
        dict: Dictionary with 'scaler', 'salary_encoder', 'department_encoder', 'feature_cols'
    """
    if models_dir is None:
        models_dir = MODELS_DIR

    return joblib.load(models_dir / "preprocessing_artifacts.pkl")
