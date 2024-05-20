"""Generate univariate analysis plots and statistics for the Predictables app."""

import streamlit as st

# Initialize state variables if needed
if "data" not in st.session_state:
    st.session_state["data"] = None

st.markdown("# Univariate Analysis")

col1, col2 = st.columns(2)