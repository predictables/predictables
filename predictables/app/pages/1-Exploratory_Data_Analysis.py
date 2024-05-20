"""Generate the EDA page of the Predictables app, using the pygwalker library."""

import streamlit as st
from pygwalker.api.streamlit import StreamlitRenderer

# Initialize state variables if needed
if "data" not in st.session_state:
    st.session_state["data"] = None


st.set_page_config(page_title="Exploratory Data Analysis", layout="wide")

st.markdown("# Exploratory Data Analysis")


data = st.session_state.data

if data is not None:
    # Generate the EDA report
    pyg_app = StreamlitRenderer(data)
    pyg_app.explorer()
else:
    st.write("No data has been loaded yet.")
    st.write("Please load data on the Load Data page.")