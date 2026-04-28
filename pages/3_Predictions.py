"""
Attrition Prediction Page
Uses trained model to predict employee attrition risk
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(page_title="Predictions", layout="wide")

@st.cache_resource
def load_model_artifacts():
    """Load trained model and preprocessing artifacts."""
    models_dir = Path("models")

    try:
        rf_model = joblib.load(models_dir / "random_forest_model.pkl")
        lr_model = joblib.load(models_dir / "logistic_regression_model.pkl")
        scaler = joblib.load(models_dir / "scaler.pkl")
        le_dept = joblib.load(models_dir / "label_encoder_department.pkl")
        le_salary = joblib.load(models_dir / "label_encoder_salary.pkl")

        with open(models_dir / "feature_columns.json", "r") as f:
            feature_cols = json.load(f)

        feature_importance = pd.read_csv(models_dir / "feature_importance.csv")

        with open(models_dir / "model_results.json", "r") as f:
            model_results = json.load(f)

        return {
            "rf_model": rf_model,
            "lr_model": lr_model,
            "scaler": scaler,
            "le_dept": le_dept,
            "le_salary": le_salary,
            "feature_cols": feature_cols,
            "feature_importance": feature_importance,
            "model_results": model_results,
        }
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

st.title("Attrition Prediction")

artifacts = load_model_artifacts()

if artifacts is None:
    st.warning("Please run the training script first: `python scripts/train_model.py`")
    st.stop()

# Model selection
col1, col2 = st.columns([2, 1])
with col1:
    selected_model = st.selectbox(
        "Select Model",
        ["Random Forest (Recommended)", "Logistic Regression"],
        index=0
    )

# Show model performance
with col2:
    model_key = "random_forest" if "Random Forest" in selected_model else "logistic_regression"
    metrics = artifacts["model_results"][model_key]
    st.metric("Accuracy", f"{metrics['accuracy']:.1%}")
    st.metric("ROC AUC", f"{metrics['roc_auc']:.3f}")

st.divider()

# Two columns: Input form and Feature importance
col_input, col_importance = st.columns([1, 1])

with col_input:
    st.subheader("Employee Information")

    with st.form("prediction_form"):
        satisfaction = st.slider(
            "Satisfaction Level",
            0.0, 1.0, 0.6, 0.01,
            help="Employee satisfaction score (0-1)"
        )

        last_evaluation = st.slider(
            "Last Evaluation Score",
            0.0, 1.0, 0.7, 0.01,
            help="Last performance evaluation (0-1)"
        )

        num_projects = st.slider(
            "Number of Projects",
            2, 7, 4,
            help="Current number of projects assigned"
        )

        monthly_hours = st.slider(
            "Average Monthly Hours",
            96, 310, 200,
            help="Average hours worked per month"
        )

        tenure = st.slider(
            "Time at Company (years)",
            2, 10, 3,
            help="Years spent at the company"
        )

        work_accident = st.selectbox(
            "Work Accident in Last Year?",
            [0, 1],
            format_func=lambda x: "No" if x == 0 else "Yes"
        )

        promotion = st.selectbox(
            "Promotion in Last 5 Years?",
            [0, 1],
            format_func=lambda x: "No" if x == 0 else "Yes"
        )

        department = st.selectbox(
            "Department",
            artifacts["le_dept"].classes_
        )

        salary = st.selectbox(
            "Salary Level",
            artifacts["le_salary"].classes_
        )

        submitted = st.form_submit_button("Predict Attrition Risk", width='stretch')

with col_importance:
    st.subheader("Feature Importance (Random Forest)")
    fig_importance = px.bar(
        artifacts["feature_importance"].head(5),
        x="importance",
        y="feature",
        orientation="h",
        title="Top 5 Predictive Features",
        color="importance",
        color_continuous_scale="Blues",
    )
    fig_importance.update_layout(showlegend=False, yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig_importance, width='stretch')

    st.caption("These features have the strongest influence on attrition predictions.")

# Handle prediction
if submitted:
    # Encode categorical inputs
    dept_encoded = artifacts["le_dept"].transform([department])[0]
    salary_encoded = artifacts["le_salary"].transform([salary])[0]

    # Create feature array
    features = np.array([[
        satisfaction, last_evaluation, num_projects, monthly_hours,
        tenure, work_accident, promotion, dept_encoded, salary_encoded
    ]])

    # Get feature names order
    feature_order = artifacts["feature_cols"]

    # Select model
    if "Random Forest" in selected_model:
        model = artifacts["rf_model"]
        proba = model.predict_proba(features)[0]
    else:
        model = artifacts["lr_model"]
        features_scaled = artifacts["scaler"].transform(features)
        proba = model.predict_proba(features_scaled)[0]

    attrition_prob = proba[1] * 100
    stay_prob = proba[0] * 100

    # Display results
    st.divider()
    st.subheader("Prediction Result")

    col_result1, col_result2, col_result3 = st.columns(3)

    with col_result1:
        risk_level = "High" if attrition_prob > 50 else "Medium" if attrition_prob > 20 else "Low"
        risk_color = "(High)" if risk_level == "High" else "(Medium)" if risk_level == "Medium" else "(Low)"

        st.metric(
            "Attrition Risk",
            f"{attrition_prob:.1f}%",
            delta=None,
            help=f"Probability this employee will leave: {attrition_prob:.1f}%"
        )

    with col_result2:
        st.metric(
            "Risk Level",
            f"{risk_level}",
            help="Based on prediction threshold"
        )

    with col_result3:
        st.metric(
            "Retention Probability",
            f"{stay_prob:.1f}%",
            help=f"Probability this employee will stay: {stay_prob:.1f}%"
        )

    # Probability gauge
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=attrition_prob,
        domain={"x": [0, 1], "y": [0, 1]},
        title={"text": "Attrition Probability"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "darkblue"},
            "steps": [
                {"range": [0, 20], "color": "lightgreen"},
                {"range": [20, 50], "color": "yellow"},
                {"range": [50, 100], "color": "lightcoral"},
            ],
            "threshold": {
                "line": {"color": "red", "width": 4},
                "thickness": 0.75,
                "value": 50
            }
        }
    ))
    fig_gauge.update_layout(height=300)
    st.plotly_chart(fig_gauge, width='stretch')

    # Recommendations based on risk factors
    st.subheader("Risk Factor Analysis")

    risk_factors = []
    if satisfaction < 0.5:
        risk_factors.append("Low satisfaction score - consider engagement initiatives")
    if tenure > 5:
        risk_factors.append("Long tenure may indicate stagnation - consider career development")
    if num_projects < 3:
        risk_factors.append("Low project count - employee may be underutilized")
    if num_projects > 6:
        risk_factors.append("High project count - potential burnout risk")
    if monthly_hours > 250:
        risk_factors.append("High working hours - burnout risk")
    if promotion == 0:
        risk_factors.append("No recent promotion - career growth concern")

    if risk_factors:
        for factor in risk_factors:
            st.info(factor)
    else:
        st.success("No significant risk factors identified")

st.divider()

# Model info section
with st.expander("About the Models"):
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        **Random Forest (Recommended)**
        - 98.3% accuracy, 99.15% ROC AUC
        - Ensemble of 100 decision trees
        - Handles non-linear relationships well
        - Best for accurate predictions
        """)

    with col2:
        st.markdown("""
        **Logistic Regression**
        - 77.1% accuracy, 81.15% ROC AUC
        - Linear model with interpretable coefficients
        - Faster predictions
        - Better for understanding feature impact
        """)
