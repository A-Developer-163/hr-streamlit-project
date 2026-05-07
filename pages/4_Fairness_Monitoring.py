"""
Fairness Monitoring Dashboard
Monitoring model fairness across salary levels and departments with Equalized Odds metrics
"""

import streamlit as st
import pandas as pd
import json
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from config import MODELS_DIR

st.set_page_config(page_title="Fairness Monitoring", layout="wide")


# Default thresholds
DEFAULT_THRESHOLDS = {
    "equalized_odds_max": 0.05,
    "demographic_parity_min": 0.8,
    "accuracy_min": 0.95
}


@st.cache_data(ttl=300)
def load_fairness_data():
    """Loading fairness metrics from JSON reports."""
    models_dir = MODELS_DIR

    data = {
        "fairness_report": {},
        "mitigation_comparison": {},
        "mitigation_report": {},
        "model_results": {},
        "loaded": False,
        "error": None
    }

    try:
        # Load all JSON files efficiently
        json_files = {
            "fairness_report.json": "fairness_report",
            "mitigation_comparison.json": "mitigation_comparison",
            "mitigation_report.json": "mitigation_report",
            "model_results.json": "model_results"
        }

        for filename, key in json_files.items():
            file_path = models_dir / filename
            try:
                with open(file_path, "r") as f:
                    data[key] = json.load(f)
            except FileNotFoundError:
                if key in ["fairness_report", "model_results"]:
                    data["error"] = f"{filename} not found. Run training script first."

        data["loaded"] = bool(data["fairness_report"] or data["mitigation_comparison"])

    except Exception as e:
        data["error"] = f"Error loading fairness data: {str(e)}"

    return data


@st.cache_data(ttl=300)
def load_thresholds():
    """Loading fairness thresholds from config file."""
    models_dir = MODELS_DIR
    thresholds_path = models_dir / "fairness_thresholds.json"

    if thresholds_path.exists():
        try:
            with open(thresholds_path, "r") as f:
                return json.load(f)
        except Exception:
            pass

    return DEFAULT_THRESHOLDS.copy()


def save_thresholds(thresholds):
    """Saving thresholds to config file."""
    models_dir = MODELS_DIR
    thresholds_path = models_dir / "fairness_thresholds.json"

    try:
        with open(thresholds_path, "w") as f:
            json.dump(thresholds, f, indent=2)
        return True
    except Exception as e:
        st.error(f"Error saving thresholds: {e}")
        return False


def get_key_metrics(fairness_data, thresholds):
    """Extracting key metrics for dashboard display."""
    metrics = {
        "worst_eq_odds": None,
        "worst_dp_ratio": None,
        "best_accuracy": None,
        "num_models": 0,
        "violations": []
    }

    # Check mitigation_comparison first (has baseline vs mitigated)
    if fairness_data["mitigation_comparison"]:
        for model_name, model_data in fairness_data["mitigation_comparison"].items():
            metrics["num_models"] += 1
            for technique, tech_data in model_data.items():
                eq_odds = tech_data.get("Eq Odds Diff", float("inf"))
                dp_ratio = tech_data.get("DP Ratio", 0)
                accuracy = tech_data.get("Accuracy", 0)

                # Track worst/best values
                if metrics["worst_eq_odds"] is None or eq_odds > metrics["worst_eq_odds"]:
                    metrics["worst_eq_odds"] = eq_odds
                if metrics["worst_dp_ratio"] is None or dp_ratio < metrics["worst_dp_ratio"]:
                    metrics["worst_dp_ratio"] = dp_ratio
                if metrics["best_accuracy"] is None or accuracy > metrics["best_accuracy"]:
                    metrics["best_accuracy"] = accuracy

                # Check for violations
                if eq_odds > thresholds["equalized_odds_max"]:
                    metrics["violations"].append({
                        "model": f"{model_name} ({technique})",
                        "metric": "Equalized Odds Difference",
                        "value": eq_odds,
                        "threshold": thresholds["equalized_odds_max"]
                    })
                if dp_ratio < thresholds["demographic_parity_min"]:
                    metrics["violations"].append({
                        "model": f"{model_name} ({technique})",
                        "metric": "Demographic Parity Ratio",
                        "value": dp_ratio,
                        "threshold": thresholds["demographic_parity_min"]
                    })
                if accuracy < thresholds["accuracy_min"]:
                    metrics["violations"].append({
                        "model": f"{model_name} ({technique})",
                        "metric": "Accuracy",
                        "value": accuracy,
                        "threshold": thresholds["accuracy_min"]
                    })

    # Fallback to fairness_report if no mitigation_comparison
    elif fairness_data["fairness_report"]:
        for model_name, model_data in fairness_data["fairness_report"].items():
            metrics["num_models"] += 1
            salary_data = model_data.get("by_salary", {})
            eq_odds = salary_data.get("equalized_odds_difference", float("inf"))
            dp_ratio = salary_data.get("demographic_parity_ratio", 0)

            if metrics["worst_eq_odds"] is None or eq_odds > metrics["worst_eq_odds"]:
                metrics["worst_eq_odds"] = eq_odds
            if metrics["worst_dp_ratio"] is None or dp_ratio < metrics["worst_dp_ratio"]:
                metrics["worst_dp_ratio"] = dp_ratio

            # Get accuracy from model_results if available
            if fairness_data["model_results"]:
                accuracy = fairness_data["model_results"].get(model_name, {}).get("accuracy", 0)
                if metrics["best_accuracy"] is None or accuracy > metrics["best_accuracy"]:
                    metrics["best_accuracy"] = accuracy

    return metrics


st.title("Fairness Monitoring Dashboard")
st.markdown("Monitoring model fairness across salary levels and departments")

# Load data with spinner
with st.spinner("Loading fairness metrics..."):
    fairness_data = load_fairness_data()

if not fairness_data["loaded"]:
    st.error(f"**No fairness data found**\n\n{fairness_data['error']}")
    st.info("Run the training script with fairness analysis:")
    st.code("python scripts/train_advanced_models.py --mitigate", language="bash")
    st.stop()

# Sidebar: Configurable thresholds
st.sidebar.header("Fairness Thresholds")

thresholds = load_thresholds()

with st.sidebar.expander("Configure Thresholds", expanded=False):
    eq_odds_max = st.number_input(
        "Max Equalized Odds Difference",
        min_value=0.0,
        max_value=1.0,
        value=thresholds["equalized_odds_max"],
        step=0.01,
        format="%.3f",
        help="Maximum acceptable difference in TPR/FPR across groups"
    )

    dp_ratio_min = st.number_input(
        "Min Demographic Parity Ratio",
        min_value=0.0,
        max_value=1.0,
        value=thresholds["demographic_parity_min"],
        step=0.05,
        format="%.2f",
        help="Minimum acceptable ratio of selection rates (EEOC 80% rule)"
    )

    accuracy_min = st.number_input(
        "Min Accuracy",
        min_value=0.0,
        max_value=1.0,
        value=thresholds["accuracy_min"],
        step=0.01,
        format="%.2f",
        help="Minimum acceptable model accuracy"
    )

    new_thresholds = {
        "equalized_odds_max": eq_odds_max,
        "demographic_parity_min": dp_ratio_min,
        "accuracy_min": accuracy_min
    }

    if st.button("Save Thresholds"):
        if save_thresholds(new_thresholds):
            st.success("Thresholds saved!")
            st.rerun()

# Update thresholds if changed
thresholds = new_thresholds

# Calculate key metrics
key_metrics = get_key_metrics(fairness_data, thresholds)

# Key Metrics Section
st.subheader("Key Fairness Metrics")
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    eq_odds = key_metrics["worst_eq_odds"]
    eq_odds_color = "normal" if eq_odds <= thresholds["equalized_odds_max"] else "inverse"
    st.metric(
        "Worst Eq Odds Diff",
        f"{eq_odds:.4f}" if eq_odds is not None else "N/A",
        help="Equalized Odds Difference (lower is better)"
    )

with col2:
    dp_ratio = key_metrics["worst_dp_ratio"]
    st.metric(
        "Worst DP Ratio",
        f"{dp_ratio:.4f}" if dp_ratio is not None else "N/A",
        help="Demographic Parity Ratio (higher is better)"
    )

with col3:
    accuracy = key_metrics["best_accuracy"]
    st.metric(
        "Best Accuracy",
        f"{accuracy:.2%}" if accuracy is not None else "N/A",
        help="Best model accuracy across all models"
    )

with col4:
    st.metric("Models Evaluated", key_metrics["num_models"])

with col5:
    num_violations = len(key_metrics["violations"])
    delta_color = "normal" if num_violations == 0 else "inverse"
    st.metric("Threshold Violations", num_violations, delta_color=delta_color)

st.divider()

# Threshold Alerts Section
if key_metrics["violations"]:
    st.subheader("Threshold Violations")

    # Extract unique model names from violations
    model_names = sorted(set(v["model"] for v in key_metrics["violations"]))

    # Find default index (lightgbm (Baseline))
    default_model = "lightgbm (Baseline)"
    options = ["All Models"] + model_names
    default_index = 0
    if default_model in options:
        default_index = options.index(default_model)

    # Model selector dropdown
    selected_model = st.selectbox(
        "Select Model to View Violations",
        options,
        index=default_index,
        help="Choose a specific model to view its threshold violations"
    )

    # Filter violations based on selection
    if selected_model == "All Models":
        filtered_violations = key_metrics["violations"]
    else:
        filtered_violations = [v for v in key_metrics["violations"] if v["model"] == selected_model]

    # Display filtered violations
    if filtered_violations:
        for violation in filtered_violations:
            st.error(
                f"**{violation['model']}**: {violation['metric']} = "
                f"{violation['value']:.4f} (threshold: {violation['threshold']:.4f})"
            )
    else:
        st.success(f"No violations found for {selected_model}")
else:
    st.success("All models within acceptable fairness thresholds")

st.divider()

# Model Comparison Charts
if fairness_data["mitigation_comparison"]:
    st.subheader("Model Comparison: Baseline vs Mitigated")

    # Prepare data for visualization
    comparison_data = []
    for model_name, model_data in fairness_data["mitigation_comparison"].items():
        for technique, tech_data in model_data.items():
            comparison_data.append({
                "Model": model_name.title(),
                "Technique": technique,
                "Accuracy": tech_data.get("Accuracy", 0),
                "Eq Odds Diff": tech_data.get("Eq Odds Diff", 0),
                "DP Ratio": tech_data.get("DP Ratio", 0)
            })

    df_comparison = pd.DataFrame(comparison_data)

    # Accuracy Comparison
    col1, col2, col3 = st.columns(3)

    with col1:
        fig_acc = px.bar(
            df_comparison,
            x="Model",
            y="Accuracy",
            color="Technique",
            barmode="group",
            title="Accuracy Comparison",
            height=400
        )
        fig_acc.add_hline(
            y=thresholds["accuracy_min"],
            line_dash="dash",
            line_color="red",
            annotation_text=f"Min: {thresholds['accuracy_min']:.2f}"
        )
        fig_acc.update_yaxes(range=[0.9, 1.0])
        st.plotly_chart(fig_acc, width="stretch")

    with col2:
        fig_eq = px.bar(
            df_comparison,
            x="Model",
            y="Eq Odds Diff",
            color="Technique",
            barmode="group",
            title="Equalized Odds Difference",
            height=400
        )
        fig_eq.add_hline(
            y=thresholds["equalized_odds_max"],
            line_dash="dash",
            line_color="green",
            annotation_text=f"Target: {thresholds['equalized_odds_max']:.3f}"
        )
        st.plotly_chart(fig_eq, width="stretch")

    with col3:
        fig_dp = px.bar(
            df_comparison,
            x="Model",
            y="DP Ratio",
            color="Technique",
            barmode="group",
            title="Demographic Parity Ratio",
            height=400
        )
        fig_dp.add_hline(
            y=thresholds["demographic_parity_min"],
            line_dash="dash",
            line_color="green",
            annotation_text=f"EEOC 80%: {thresholds['demographic_parity_min']:.2f}"
        )
        st.plotly_chart(fig_dp, width="stretch")

st.divider()

# Per-Group Breakdown
st.subheader("Per-Group Breakdown")

col1, col2 = st.columns([1, 3])

with col1:
    sensitive_attr = st.selectbox(
        "Sensitive Attribute",
        ["Salary", "Department"],
        help="Choose demographic group to analyse"
    )

    # Determine available models
    available_models = []
    if fairness_data["mitigation_comparison"]:
        available_models = list(fairness_data["mitigation_comparison"].keys())
    elif fairness_data["fairness_report"]:
        available_models = list(fairness_data["fairness_report"].keys())

    selected_model = st.selectbox(
        "Select Model",
        available_models,
        help="Choose model to analyse"
    )

with col2:
    if fairness_data["fairness_report"] and selected_model:
        model_data = fairness_data["fairness_report"].get(selected_model, {})

        if sensitive_attr == "Salary":
            attr_key = "by_salary"
            group_order = ["high", "medium", "low"]
        else:  # Department
            attr_key = "by_department"
            # Will get groups from data
            group_order = []

        group_data = model_data.get(attr_key, {}).get("by_group", {})

        if group_data:
            # Transform data structure: by_group has {metric: {group: value}}
            # We need: {group: {metric: value}}
            # Get all unique group names from any metric
            if not group_order:
                # Get groups from the first metric that has groups
                for metric_name, metric_values in group_data.items():
                    if isinstance(metric_values, dict):
                        group_order = sorted(metric_values.keys())
                        break

            # Prepare per-group metrics
            group_metrics = []
            for group in group_order:
                metrics_dict = {}
                # Collect all metrics for this group
                for metric_name, metric_values in group_data.items():
                    if isinstance(metric_values, dict) and group in metric_values:
                        metrics_dict[metric_name] = metric_values[group]

                if metrics_dict:
                    group_metrics.append({
                        "Group": group.title(),
                        "TPR": metrics_dict.get("true_positive_rate", 0),
                        "FPR": metrics_dict.get("false_positive_rate", 0),
                        "Selection Rate": metrics_dict.get("selection_rate", 0),
                        "Accuracy": metrics_dict.get("accuracy", 0)
                    })

            df_groups = pd.DataFrame(group_metrics)

            if not df_groups.empty and "Group" in df_groups.columns:
                # Create grouped bar chart
                fig_groups = go.Figure()

                fig_groups.add_trace(go.Bar(
                    name="True Positive Rate",
                    x=df_groups["Group"],
                    y=df_groups["TPR"],
                    marker_color="steelblue"
                ))

                fig_groups.add_trace(go.Bar(
                    name="False Positive Rate",
                    x=df_groups["Group"],
                    y=df_groups["FPR"],
                    marker_color="coral"
                ))

                fig_groups.update_layout(
                    title=f"{sensitive_attr} Level: TPR and FPR by Group",
                    barmode="group",
                    xaxis_title=f"{sensitive_attr} Level",
                    yaxis_title="Rate",
                    height=400
                )

                st.plotly_chart(fig_groups, width="stretch")

                # Detailed metrics table
                with st.expander("View Detailed Metrics"):
                    st.dataframe(df_groups, width="stretch", hide_index=True)
            else:
                st.warning(f"No per-group metrics available for {selected_model}")
        else:
            st.warning(f"No {sensitive_attr.lower()} data available for {selected_model}")

st.divider()

# Information Section
st.subheader("Understanding the Metrics")

col1, col2 = st.columns(2)

with col1:
    st.info("""
**Equalized Odds Difference**
- Measures the maximum difference in True Positive Rates and False Positive Rates across groups
- **Target**: ≤0.05 (lower is better)
- Ensures the model is equally accurate across all demographic groups
""")

with col2:
    st.info("""
**Demographic Parity Ratio**
- Ratio of selection rates between the worst and best performing groups
- **EEOC 80% Rule**: ≥0.8 required for legal compliance
- Note: Low values may reflect genuine differences in attrition rates
""")

# Footer
st.divider()
st.caption("Last updated: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
st.caption("Fairness metrics computed using fairlearn library")
