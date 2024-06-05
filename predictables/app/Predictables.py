"""Define the main page of the Predictables app."""

from __future__ import annotations
import streamlit as st
import pandas as pd

from predictables.app import initialize_state
from predictables.app.src.util import update_state


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

    for column in ["fold", "target"]:
        if column in data.columns:
            data[column] = data[column].fillna(-1).astype(int)

    # add data to the state and return it
    update_state("data", data)

    return data


def main() -> None:
    """Generate the main page."""
    st.title("Predictables")
    st.write("This is a demo of the Predictables app.")
    st.write("The app is currently under construction.")
    st.write("Please check back later.")

    # Initialize state variables
    initialize_state()


if __name__ == "__main__":
    main()