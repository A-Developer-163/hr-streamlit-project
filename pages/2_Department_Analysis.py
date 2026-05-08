"""
Department Analysis Page
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from session_utils import get_hr_data

st.set_page_config(page_title="Department Analysis", layout="wide")

st.title("Department Analysis")

df = get_hr_data()

if df is not None and not df.empty:
    # Department selector
    selected_dept = st.selectbox(
        "Select Department",
        options=["All Departments"] + sorted(df["department"].unique().tolist())
    )

    if selected_dept != "All Departments":
        df_filtered = df[df["department"] == selected_dept].copy()
    else:
        df_filtered = df.copy()

    # Department summary
    dept_summary = df_filtered.groupby("department").agg({
        "employee_id": "count",
        "attrition": "sum",
        "satisfaction_level": "mean",
        "last_evaluation": "mean",
        "avg_monthly_hours": "mean",
        "num_projects": "mean",
    }).round(2)

    dept_summary.columns = ["Employees", "Left", "Avg Satisfaction", "Avg Evaluation", "Avg Hours", "Avg Projects"]
    dept_summary["Attrition Rate"] = (dept_summary["Left"] / dept_summary["Employees"] * 100).round(1)
    dept_summary = dept_summary.sort_values("Attrition Rate", ascending=False)

    st.subheader("Department Summary")
    st.dataframe(dept_summary, width='stretch')

    st.divider()

    # Charts
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Department Size Distribution")
        fig_size = px.pie(
            df_filtered.groupby("department").size().reset_index(name="count"),
            values="count",
            names="department",
            title="Employee Distribution by Department"
        )
        st.plotly_chart(fig_size, width='stretch')

    with col2:
        st.subheader("Average Hours by Department")
        dept_hours = df_filtered.groupby("department")["avg_monthly_hours"].mean().sort_values(ascending=False)
        fig_hours = px.bar(
            x=dept_hours.index,
            y=dept_hours.values,
            title="Average Monthly Hours by Department",
            color=dept_hours.values,
            color_continuous_scale="Blues",
        )
        fig_hours.update_xaxes(title="Department")
        fig_hours.update_yaxes(title="Avg Monthly Hours")
        st.plotly_chart(fig_hours, width='stretch')

    # Satisfaction by Department
    st.subheader("Satisfaction Levels by Department")
    fig_sat = px.box(
        df_filtered,
        x="department",
        y="satisfaction_level",
        title="Satisfaction Level Distribution by Department",
        color="department"
    )
    fig_sat.update_layout(showlegend=False)
    st.plotly_chart(fig_sat, width='stretch')
