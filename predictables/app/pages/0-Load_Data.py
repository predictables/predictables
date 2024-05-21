"""Load data into the Predictables app."""

import streamlit as st
from predictables.app import initialize_state, is_data_loaded, update_state
from predictables.app.Predictables import load_data

# Initialize state variables if needed
initialize_state()

st.set_page_config(page_title="Load Data", layout="wide")

st.markdown("# Load Data")

# Load the data from the selected csv or parquet file dialog
st.write(
    "Upload a CSV or Parquet file to use Predictables. If multiple files are uploaded, the files will attempt to be concatenated, and if they cannot be concatenated, the first file will be used."
)

col1, col2 = st.columns(2)

col2.write(
    f"Current target variable: {st.session_state['target_variable']}",
    key="target-variable-statement",
)

file = col1.file_uploader(
    "Upload a CSV or Parquet file", type=["csv", "parquet"], accept_multiple_files=True
)


if is_data_loaded():
    # Select the target variable from the dataset to use
    target_variable = col2.selectbox(
        "Target Variable",
        st.session_state["columns"],
        key="target-variable",
        placeholder="Target variable...",
        on_change=lambda: update_state("target_variable", target_variable),
    )
    st.dataframe(st.session_state["data"])

elif file:
    df = load_data(file)  # Load the data
    st.session_state.update(data=df)  # Update the state with the data
    st.session_state.update(
        columns=df.columns.tolist()
    )  # Update the state with the columns
    target_variable = col2.selectbox(
        "Target Variable",
        st.session_state["columns"],
        key="target-variable",
        placeholder="Target variable...",
        on_change=lambda: update_state("target_variable", target_variable),
    )

    # rewrite the target variable statement
    col2.write(
        f"Current target variable: {st.session_state['target_variable']}",
        key="target-variable-statement",
    )
    st.dataframe(st.session_state["data"])