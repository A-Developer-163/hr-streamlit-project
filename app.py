"""
HR Employee Analytics - Main Dashboard
Interactive EDA with Attrition Prediction
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="HR Employee Analytics",
    layout="wide",
)

@st.cache_data
def load_data():
    """Load HR employee data from CSV."""
    try:
        df = pd.read_csv("data/hr_employee_data.csv")
        for col in df.select_dtypes(include=["object", "string"]).columns:
            df[col] = df[col].astype("category")
        return df
    except FileNotFoundError:
        st.error("CSV file not found. Run: python scripts/transform_data.py")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

@st.cache_data
def get_summary_stats(_df):
    """Calculate summary statistics."""
    total = len(_df)
    left = _df["left"].sum()
    stayed = total - left
    attrition_rate = (left / total) * 100
    avg_satisfaction = _df["satisfaction_level"].mean()
    avg_hours = _df["average_montly_hours"].mean()

    return {
        "total": total,
        "left": left,
        "stayed": stayed,
        "attrition_rate": attrition_rate,
        "avg_satisfaction": avg_satisfaction,
        "avg_hours": avg_hours,
        "departments": _df["Department"].nunique(),
    }

def main():
    st.title("HR Employee Analytics Dashboard")

    df = load_data()

    if df.empty:
        st.warning("Unable to load data. Please ensure `data/hr_employee_data.csv` exists.")
        return

    stats = get_summary_stats(df)

    # Key Metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Total Employees", f"{stats['total']:,}")
    with col2:
        st.metric("Active", f"{stats['stayed']:,}")
    with col3:
        st.metric("Left", f"{stats['left']:,}")
    with col4:
        st.metric("Attrition Rate", f"{stats['attrition_rate']:.1f}%")
    with col5:
        st.metric("Departments", stats['departments'])

    st.divider()

    # Two column layout for charts
    col_left, col_right = st.columns(2)

    # Attrition by Department
    with col_left:
        st.subheader("Attrition by Department")
        dept_attrition = df.groupby("Department")["left"].agg(["sum", "count"]).reset_index()
        dept_attrition["attrition_pct"] = (dept_attrition["sum"] / dept_attrition["count"] * 100).round(1)
        dept_attrition = dept_attrition.sort_values("attrition_pct", ascending=False)

        fig_dept = px.bar(
            dept_attrition,
            x="attrition_pct",
            y="Department",
            orientation="h",
            title="Attrition Rate by Department (%)",
            color="attrition_pct",
            color_continuous_scale="RdYlGn_r",
        )
        fig_dept.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig_dept, width='stretch')

    # Satisfaction Distribution
    with col_right:
        st.subheader("Satisfaction Distribution")
        fig_sat = px.histogram(
            df,
            x="satisfaction_level",
            nbins=20,
            title="Satisfaction Level Distribution",
            color="left",
            color_discrete_map={0: "#2ecc71", 1: "#e74c3c"},
        )
        fig_sat.update_layout(bargap=0.1)
        st.plotly_chart(fig_sat, width='stretch')

    # Second row of charts
    col_left2, col_right2 = st.columns(2)

    # Salary vs Attrition
    with col_left2:
        st.subheader("Attrition by Salary Level")
        salary_attrition = df.groupby("salary")["left"].mean() * 100
        fig_salary = px.bar(
            x=salary_attrition.index,
            y=salary_attrition.values,
            title="Attrition Rate by Salary Level (%)",
            color=salary_attrition.values,
            color_continuous_scale="Reds",
        )
        fig_salary.update_xaxes(title="Salary Level")
        fig_salary.update_yaxes(title="Attrition Rate (%)")
        st.plotly_chart(fig_salary, width='stretch')

    # Monthly Hours Distribution
    with col_right2:
        st.subheader("Monthly Hours by Status")
        fig_hours = px.box(
            df,
            x="left",
            y="average_montly_hours",
            title="Monthly Hours: Stayed vs Left",
            color="left",
            color_discrete_map={0: "#2ecc71", 1: "#e74c3c"},
        )
        fig_hours.update_xaxes(ticktext=["Stayed", "Left"], tickvals=[0, 1])
        st.plotly_chart(fig_hours, width='stretch')

    st.divider()

    # Key Insights
    st.subheader("Key Insights")
    col1, col2 = st.columns(2)

    with col1:
        st.info(f"""
        **Attrition Overview:**
        - {stats['left']:,} employees left ({stats['attrition_rate']:.1f}% rate)
        - {stats['stayed']:,} employees remain active
        - Average satisfaction: {stats['avg_satisfaction']:.2f}/1.0
        """)

    with col2:
        st.info(f"""
        **Working Patterns:**
        - Average monthly hours: {stats['avg_hours']:.0f}
        - {stats['departments']} departments tracked
        - Salary distribution: Low (48.8%), Medium (43.0%), High (8.2%)
        """)

if __name__ == "__main__":
    main()
