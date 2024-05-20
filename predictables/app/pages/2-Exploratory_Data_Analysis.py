"""Generate the EDA page of the Predictables app, using the pygwalker library."""

import streamlit as st
from pygwalker.api.streamlit import StreamlitRenderer
import pandas as pd

st.set_page_config(page_title="Exploratory Data Analysis", layout="wide")

st.markdown("# Exploratory Data Analysis")

# Load the data from the selected csv or parquet file dialog
st.write(
    "Upload a CSV or Parquet file to perform exploratory data analysis. If multiple files are uploaded, the files will attempt to be concatenated, and if they cannot be concatenated, the first file will be used."
)
file = st.file_uploader(
    "Upload a CSV or Parquet file", type=["csv", "parquet"], accept_multiple_files=True
)

if file:
    # Try to load all data sets and concatenate them, but if that fails, just load the first data set
    dflist = [
        pd.read_csv(f) if f.name.endswith(".csv") else pd.read_parquet(f) for f in file
    ]
    try:
        data = pd.concat(dflist, ignore_index=True).reset_index(drop=True)
    except ValueError as _:
        data = dflist[0].reset_index(drop=True)

    # Display the data
    st.dataframe(data)

    # Generate the EDA report
    pyg_app = StreamlitRenderer(data)
    pyg_app.explorer()
