"""Generate the EDA page of the Predictables app, using the pygwalker library."""

import streamlit as st
from pygwalker.api.streamlit import StreamlitRenderer
from predictables.app import initialize_state
from predictables.app.src.util import is_data_loaded, get_data

# Initialize state variables if needed
initialize_state()

st.set_page_config(page_title="Exploratory Data Analysis", layout="wide")

st.markdown("# Exploratory Data Analysis")

if is_data_loaded():
    # Generate the EDA report
    pyg_app = StreamlitRenderer(get_data())
    pyg_app.explorer()
else:
    st.write("No data has been loaded yet.")
    st.write("Please load data on the Load Data page.")