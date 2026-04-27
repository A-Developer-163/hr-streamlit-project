"""
Overview Page - Employee Demographics
"""

import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Overview", page_icon="📊")

@st.cache_data
def load_data():
    return pd.read_excel("data/hr_employee_data.xlsx")

st.title("📊 Employee Overview")

df = load_data()

# Display basic info
st.subheader("Dataset Information")
st.write(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")
st.write("**Data Types:**")
st.dataframe(df.dtypes.to_frame('Data Type'))
