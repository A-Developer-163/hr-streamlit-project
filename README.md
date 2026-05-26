## Project Overview

This is an Interactive HR analytics dashboard built with Streamlit, that explores employee attrition rates.

**Data Source**: https://www.kaggle.com/datasets/kmldas/hr-employee-data-descriptive-analytics

## Features

- **Interactive Dashboard**: HR metrics with key indicators (total employees, attrition rate, departments)
- **Multi-Page App**:
  - Overview: Employee demographics and dataset information
  - Attrition Analysis: Factors contributing to employee turnover
  - Department Analysis: Department-level breakdowns and insights
  - Predictions: ML-based attrition predictions with trained models
  - Fairness Monitoring: Model fairness metrics and bias mitigation
- **Visualisations**: Department distributions, satisfaction levels, salary analysis, working hours
- **ML-Powered Prediction models**: Trained models (XGBoost, LightGBM), SHAP explainability, and fairness monitoring

## Tech Stack

- **Frontend**: Streamlit
- **Backend**: Python (pandas, seaborn, plotly)
- **ML/Data Science**: scikit-learn, XGBoost, LightGBM, SHAP, fairlearn
- **Notebooks**: Jupyter
- **Testing**: pytest

## Installation

### Using Docker

```bash
# Development mode (live code reloading)
docker-compose up development

# Production mode
docker-compose up streamlit
```

### Local (Manual) Development

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install in editable mode
pip install -e .

# Install dev dependencies
pip install -e ".[dev]"

# Run the app
streamlit run app.py
```

## Development

```bash
# Run tests
pytest

# Format code
ruff format .

# Check linting
ruff check .
```

## License

This project is licensed under the terms of the MIT license.
