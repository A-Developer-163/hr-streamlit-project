"""
Shared session state management for HR Analytics Streamlit app.
Provides common data and utilities across all pages.
"""

import streamlit as st
import pandas as pd
from config import HR_DATA_PATH


def init_session_state():
    """Initializing shared session state data."""
    # Initialise HR data if not already loaded
    if "hr_data" not in st.session_state:
        with st.spinner("Loading HR data..."):
            st.session_state.hr_data = load_hr_data()

    # Initialise summary stats if not already calculated
    if "summary_stats" not in st.session_state and st.session_state.hr_data is not None:
        st.session_state.summary_stats = calculate_summary_stats(st.session_state.hr_data)

    # Track if data has been loaded
    if "data_loaded" not in st.session_state:
        st.session_state.data_loaded = st.session_state.hr_data is not None


@st.cache_data
def load_hr_data():
    """Loading HR employee data from CSV."""
    try:
        df = pd.read_csv(HR_DATA_PATH)
        for col in df.select_dtypes(include=["object", "string"]).columns:
            df[col] = df[col].astype("category")
        return df
    except FileNotFoundError:
        return None
    except Exception as e:
        return None


def calculate_summary_stats(df):
    """Calculating summary statistics from HR data."""
    if df is None or df.empty:
        return {}

    total = len(df)
    left = df["attrition"].sum()
    stayed = total - left
    attrition_rate = (left / total) * 100 if total > 0 else 0
    avg_satisfaction = df["satisfaction_level"].mean()
    avg_hours = df["avg_monthly_hours"].mean()

    return {
        "total": total,
        "left": left,
        "stayed": stayed,
        "attrition_rate": attrition_rate,
        "avg_satisfaction": avg_satisfaction,
        "avg_hours": avg_hours,
        "departments": df["department"].nunique(),
    }


def get_hr_data():
    """Getting HR data from session state."""
    init_session_state()
    return st.session_state.hr_data


def get_summary_stats():
    """Getting summary statistics from session state."""
    init_session_state()
    return st.session_state.summary_stats
