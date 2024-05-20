"""Define the main page of the Predictables app."""

from __future__ import annotations
import streamlit as st
import pandas as pd


@st.cache_data
def load_data(file: list[str] | None = None) -> pd.DataFrame | None:
    """Load data from a CSV or Parquet file, or from a set of files."""
    if not file:
        return None

    dflist = [
        pd.read_csv(f) if f.name.endswith(".csv") else pd.read_parquet(f) for f in file
    ]
    # Try to load all data sets and concatenate them, but if that fails, just load the first data set
    try:
        data = pd.concat(dflist, ignore_index=True).reset_index(drop=True)
    except ValueError as _:
        data = dflist[0].reset_index(drop=True)

    # add data to the state and return it
    st.session_state["data"] = data

    return data


def main() -> None:
    """Generate the main page."""
    st.title("Predictables")
    st.write("This is a demo of the Predictables app.")
    st.write("The app is currently under construction.")
    st.write("Please check back later.")

    # Initialize state variables
    if "data" not in st.session_state:
        st.session_state["data"] = None

    if "columns" not in st.session_state:
        st.session_state["columns"] = None


if __name__ == "__main__":
    main()