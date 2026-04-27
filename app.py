"""
HR Employee Descriptive Analytics - Main Application
"""

import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(
    page_title="HR Employee Analytics",
    page_icon="👥",
    layout="wide",
)

@st.cache_data
def load_data():
    """Load HR employee data from Transformed CSV."""
    try:
        df = pd.read_csv("data/hr_employee_data.csv")
        # Optimize data types
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
def get_data_summary(_df):
    """Cache expensive summary calculations."""
    return {
        "total": len(_df),
        "columns": len(_df.columns),
        "missing": _df.isnull().sum().sum(),
        "column_list": list(_df.columns),
    }

st.title("👥 HR Employee Descriptive Analytics")

df = load_data()

if not df.empty:
    summary = get_data_summary(df)
    st.sidebar.success(f"Loaded {summary['total']:,} employee records")

    # Show basic stats (cached)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Employees", f"{summary['total']:,}")
    with col2:
        st.metric("Columns", summary['columns'])
    with col3:
        st.metric("Missing Values", summary['missing'])

    # Show data preview - use height to limit rendering
    st.subheader("Data Preview")
    st.dataframe(df.head(10), width="stretch", height=400)

    # Show columns
    st.subheader("Available Columns")
    st.write(summary['column_list'])
else:
    st.warning("Unable to load data. Please ensure `data/hr_employee_data.xlsx` exists.")
