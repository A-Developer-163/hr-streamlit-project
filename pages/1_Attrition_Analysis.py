"""
Detailed Attrition Analysis Page
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from session_utils import get_hr_data

st.set_page_config(page_title="Attrition Analysis", layout="wide")

st.title("Detailed Attrition Analysis")

df = get_hr_data()

if df is not None and not df.empty:
    # Filters
    col1, col2, col3 = st.columns(3)
    with col1:
        dept_filter = st.multiselect(
            "Filter by Department",
            options=df["department"].unique(),
            default=df["department"].unique()
        )
    with col2:
        salary_filter = st.multiselect(
            "Filter by Salary",
            options=df["salary"].unique(),
            default=df["salary"].unique()
        )
    with col3:
        satisfaction_range = st.slider(
            "Satisfaction Level Range",
            min_value=float(df["satisfaction_level"].min()),
            max_value=float(df["satisfaction_level"].max()),
            value=(float(df["satisfaction_level"].min()), float(df["satisfaction_level"].max()))
        )

    # Apply filters
    filtered_df = df[
        (df["department"].isin(dept_filter)) &
        (df["salary"].isin(salary_filter)) &
        (df["satisfaction_level"].between(satisfaction_range[0], satisfaction_range[1]))
    ].copy()

    st.caption(f"Showing {len(filtered_df):,} employees (filtered from {len(df):,} total)")

    # Attrition rate for filtered data
    if len(filtered_df) > 0:
        attrition_rate = filtered_df["attrition"].mean() * 100
        st.metric("Filtered Attrition Rate", f"{attrition_rate:.1f}%")

    st.divider()

    # Analysis columns
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Tenure vs Attrition")
        tenure_attrition = filtered_df.groupby("years_at_company")["attrition"].agg(["sum", "count", "mean"]).reset_index()
        tenure_attrition["attrition_pct"] = tenure_attrition["mean"] * 100

        fig_tenure = px.line(
            tenure_attrition,
            x="years_at_company",
            y="attrition_pct",
            title="Attrition Rate by Tenure (Years)",
            markers=True,
            labels={"years_at_company": "Years at Company", "attrition_pct": "Attrition Rate (%)"}
        )
        fig_tenure.update_traces(line_color="#e74c3c", marker_size=10)
        st.plotly_chart(fig_tenure, width='stretch')

    with col2:
        st.subheader("Projects vs Attrition")
        project_attrition = filtered_df.groupby("num_projects")["attrition"].agg(["sum", "count", "mean"]).reset_index()
        project_attrition["attrition_pct"] = project_attrition["mean"] * 100

        fig_projects = px.bar(
            project_attrition,
            x="num_projects",
            y="attrition_pct",
            title="Attrition Rate by Number of Projects",
            color="attrition_pct",
            color_continuous_scale="Reds",
            labels={"num_projects": "Number of Projects", "attrition_pct": "Attrition Rate (%)"}
        )
        st.plotly_chart(fig_projects, width='stretch')

    # Second row
    col3, col4 = st.columns(2)

    with col3:
        st.subheader("Last Evaluation Score")
        eval_attrition = filtered_df.groupby(pd.cut(filtered_df["last_evaluation"], bins=5))["attrition"].mean() * 100

        fig_eval = px.bar(
            x=eval_attrition.index.astype(str),
            y=eval_attrition.values,
            title="Attrition by Last Evaluation Score",
            color=eval_attrition.values,
            color_continuous_scale="RdYlGn_r",
            labels={"x": "Evaluation Score Range", "y": "Attrition Rate (%)"}
        )
        st.plotly_chart(fig_eval, width='stretch')

    with col4:
        st.subheader("Work Accident Impact")
        accident_data = filtered_df.groupby("had_work_accident")["attrition"].agg(["count", "sum"])
        accident_data["attrition_pct"] = (accident_data["sum"] / accident_data["count"] * 100).round(1)

        fig_accident = go.Figure(data=[
            go.Bar(name="Stayed", x=accident_data.index, y=accident_data["count"] - accident_data["sum"], marker_color="#2ecc71"),
            go.Bar(name="Left", x=accident_data.index, y=accident_data["sum"], marker_color="#e74c3c")
        ])
        fig_accident.update_xaxes(ticktext=["No Accident", "Had Accident"], tickvals=[0, 1])
        fig_accident.update_layout(barmode="stack", title="Employees by Work Accident Status")
        st.plotly_chart(fig_accident, width='stretch')

    # Correlation Analysis
    st.subheader("Feature Correlation with Attrition")
    numeric_cols = ["satisfaction_level", "last_evaluation", "num_projects", "avg_monthly_hours", "years_at_company"]
    correlations = filtered_df[numeric_cols].corrwith(filtered_df["attrition"]).abs().sort_values(ascending=False)

    fig_corr = px.bar(
        x=correlations.values,
        y=correlations.index,
        orientation="h",
        title="Absolute Correlation with Attrition",
        color=correlations.values,
        color_continuous_scale="Blues",
        labels={"x": "Absolute Correlation", "y": "Feature"}
    )
    st.plotly_chart(fig_corr, width='stretch')
