"""Load data into the Predictables app."""

import streamlit as st
from predictables.app import initialize_state
from predictables.app.src.util import (
    is_data_loaded,
    update_state,
    get_data,
    get_features,
    get_excluded_features,
    exclude_feature,
)
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


file = col1.file_uploader(
    "Upload a CSV or Parquet file", type=["csv", "parquet"], accept_multiple_files=True
)


if is_data_loaded():
    pass

elif file:
    df = load_data(file)  # Load the data
    st.session_state.update(data=df)  # Update the state with the data
    st.session_state.update(
        columns=df.columns.tolist()
    )  # Update the state with the columns

col2a, col2b = col2.columns(2)

s_maybe = "s" if len(get_excluded_features()) != 1 else ""
col2a.write(f"Currently, {len(get_excluded_features())} variable{s_maybe} excluded")
feature_to_exclude = col2a.selectbox(
    "Exclude a variable", get_features(), key="exclude-variable"
)
col2a.button(
    "Exclude Variable",
    key="exclude-variable-button",
    on_click=lambda: exclude_feature(feature_to_exclude),
)

target_variable = col2b.selectbox(
    "Target Variable",
    list(set(st.session_state["columns"]) - set(get_excluded_features())),
    key="target-variable",
    placeholder="Target variable...",
    on_change=lambda: update_state("target_variable", target_variable),
)

col2b.button(
    "Update Target Variable",
    key="update-target-variable",
    on_click=lambda: update_state("target_variable", target_variable),
)

# rewrite the target variable statement
col2a.write(
    f"Current target variable: {st.session_state['target_variable']}",
    key="target-variable-statement",
)

is_time_series_data = col2b.checkbox(
    "Is the data time series data?",
    key="time-series-data",
    value=True,
    on_change=lambda: update_state("is_time_series_data", is_time_series_data),
)

if is_data_loaded():
    st.dataframe(get_data())

    st.markdown("#### Excluded Features")
    st.write(
        get_excluded_features() if get_excluded_features() else "No features excluded"
    )