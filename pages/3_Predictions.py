"""
Attrition Prediction Page
Using trained models to predict employee attrition risk
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import plotly.graph_objects as go
import plotly.express as px
from config import MODELS_DIR, load_preprocessing_artifacts_combined

st.set_page_config(page_title="Predictions", layout="wide")


@st.cache_resource
def load_preprocessing_artifacts():
    """Loading preprocessing artifacts (encoders, scalers, feature columns)."""
    try:
        artifacts = load_preprocessing_artifacts_combined()
        # Return with same keys for backward compatibility
        return {
            "scaler": artifacts["scaler"],
            "le_dept": artifacts["department_encoder"],
            "le_salary": artifacts["salary_encoder"],
            "feature_cols": artifacts["feature_cols"],
        }
    except Exception as e:
        st.error(f"Error loading preprocessing artifacts: {e}")
        return None


@st.cache_resource
def load_model(model_name):
    """Loading a single model on-demand."""
    models_dir = MODELS_DIR

    with st.spinner(f"Loading {model_name} model..."):
        try:
            if model_name == "random_forest":
                return joblib.load(models_dir / "random_forest_model.pkl")
            elif model_name == "logistic_regression":
                return joblib.load(models_dir / "lr_attrition_model.pkl")
            elif model_name == "xgboost":
                return joblib.load(models_dir / "xgboost_model.pkl")
            elif model_name == "lightgbm":
                return joblib.load(models_dir / "lightgbm_model.pkl")
        except FileNotFoundError:
            return None


@st.cache_data
def load_model_results():
    """Loading model performance metrics."""
    models_dir = MODELS_DIR

    try:
        with open(models_dir / "model_results.json", "r") as f:
            return json.load(f)
    except Exception:
        return {}


@st.cache_data
def load_feature_importance(model_name):
    """Loading feature importance for a specific model."""
    models_dir = MODELS_DIR

    try:
        return pd.read_csv(models_dir / f"feature_importance_{model_name}.csv")
    except FileNotFoundError:
        return None


st.title("Attrition Prediction")

# Load preprocessing artifacts
preprocessing = load_preprocessing_artifacts()

if preprocessing is None:
    st.warning("Please run the training script first: `python scripts/train_advanced_models.py`")
    st.stop()

# Load model results
model_results = load_model_results()

# Building model options list
model_options = ["Random Forest (Recommended)"]
if "logistic_regression" in model_results:
    model_options.append("Logistic Regression")
if "xgboost" in model_results:
    model_options.append("XGBoost")
if "lightgbm" in model_results:
    model_options.append("LightGBM")

# Model selection
col1, col2 = st.columns([2, 1])
with col1:
    selected_model = st.selectbox(
        "Select Model",
        model_options,
        index=0
    )

# Determining model key and loading metrics
model_key_map = {
    "Random Forest": "random_forest",
    "Logistic Regression": "logistic_regression",
    "XGBoost": "xgboost",
    "LightGBM": "lightgbm"
}

# Showing model performance
with col2:
    model_key = model_key_map.get(selected_model.replace(" (Recommended)", ""), "random_forest")
    if model_key in model_results:
        metrics = model_results[model_key]
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
            preprocessing["le_dept"].categories_[0]
        )

        salary = st.selectbox(
            "Salary Level",
            preprocessing["le_salary"].categories_[0]
        )

        submitted = st.form_submit_button("Predict Attrition Risk")

with col_importance:
    # Displaying feature importance for selected model (lazy-loaded in expander)
    importance_key = model_key_map.get(selected_model.replace(" (Recommended)", ""), "random_forest")
    with st.expander(f"Feature Importance ({selected_model.replace(' (Recommended)', '')})"):
        feature_importance = load_feature_importance(importance_key)
        if feature_importance is not None:
            fig_importance = px.bar(
                feature_importance.head(5),
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

# Handling prediction
if submitted:
    # Encoding categorical inputs - use DataFrames to avoid sklearn warnings
    dept_encoded = preprocessing["le_dept"].transform(
        pd.DataFrame([[department]], columns=["department"])
    )[0]
    salary_encoded = preprocessing["le_salary"].transform(
        pd.DataFrame([[salary]], columns=["salary"])
    )[0][0]

    # Creating feature array (17 features total)
    # 7 numeric + 1 salary + 9 one-hot departments
    features = np.array([[
        satisfaction, last_evaluation, num_projects, monthly_hours,
        tenure, work_accident, promotion, salary_encoded
    ] + list(dept_encoded)])

    # Selecting model and getting prediction (lazy load)
    with st.spinner("Running prediction..."):
        if "Random Forest" in selected_model:
            model = load_model("random_forest")
            proba = model.predict_proba(features)[0]
        elif "Logistic Regression" in selected_model:
            model = load_model("logistic_regression")
            # Wrap in DataFrame with proper column names
            feature_cols = preprocessing["feature_cols"]
            features_df = pd.DataFrame([features[0]], columns=feature_cols)
            features_scaled = preprocessing["scaler"].transform(features_df)
            proba = model.predict_proba(features_scaled)[0]
        elif "XGBoost" in selected_model:
            model = load_model("xgboost")
            proba = model.predict_proba(features)[0]
        elif "LightGBM" in selected_model:
            model = load_model("lightgbm")
            proba = model.predict_proba(features)[0]

    attrition_prob = proba[1] * 100
    stay_prob = proba[0] * 100

    # Displaying results
    st.divider()
    st.subheader("Prediction Result")

    col_result1, col_result2, col_result3 = st.columns(3)

    with col_result1:
        risk_level = "High" if attrition_prob > 50 else "Medium" if attrition_prob > 20 else "Low"

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
            - 98.5% accuracy, 99.2% ROC AUC
            - Ensemble of 100 decision trees
            - Handles non-linear relationships well
            - Best for accurate predictions
            """)

        if "xgboost" in model_results:
            xgb_metrics = model_results.get("xgboost", {})
            st.markdown(f"""
                **XGBoost**
                - {xgb_metrics.get('accuracy', 0):.1%} accuracy, {xgb_metrics.get('roc_auc', 0):.4f} ROC AUC
                - Gradient boosting with trees
                - Excellent performance on tabular data
                - Fast training and prediction
                """)

    with col2:
        if "lightgbm" in model_results:
            lgb_metrics = model_results.get("lightgbm", {})
            st.markdown(f"""
                **LightGBM**
                - {lgb_metrics.get('accuracy', 0):.1%} accuracy, {lgb_metrics.get('roc_auc', 0):.4f} ROC AUC
                - Light gradient boosting
                - Memory-efficient and fast
                - Handles large datasets well
                """)

        st.markdown("""
            **Logistic Regression**
            - 77.1% accuracy, 81.2% ROC AUC
            - Linear model with interpretable coefficients
            - Faster predictions
            - Better for understanding feature impact
            """)
