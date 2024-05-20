"""Load data into the Predictables app."""

import streamlit as st
from predictables.app.Predictables import load_data

# Initialize state variables if needed
if "data" not in st.session_state:
    st.session_state["data"] = None


st.set_page_config(page_title="Load Data", layout="wide")

st.markdown("# Load Data")

# Load the data from the selected csv or parquet file dialog
st.write(
    "Upload a CSV or Parquet file to use Predictables. If multiple files are uploaded, the files will attempt to be concatenated, and if they cannot be concatenated, the first file will be used."
)
file = st.file_uploader(
    "Upload a CSV or Parquet file", type=["csv", "parquet"], accept_multiple_files=True
)

if st.session_state.get("data", None) is not None:
    st.dataframe(st.session_state["data"])

elif file:
    df = load_data(file)  # Load the data
    st.session_state.update(data=df)  # Update the state with the data
    st.session_state.update(
        columns=df.columns.tolist()
    )  # Update the state with the columns
    st.dataframe(st.session_state["data"])  # Display the data
