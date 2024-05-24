"""Help the app function."""

from __future__ import annotations
import streamlit as st
import pandas as pd
import numpy as np
from typing import Any


@st.experimental_dialog("Confirm Excluded Variable")
def confirm_excluded_variable(variable: str) -> None:
    """Confirm the exclusion of a variable."""

    # Pop-up to confirm the exclusion of a variable
    def _on_click_yes(variable: str) -> None:
        """Exclude a variable."""
        exclude_feature(variable)
        st.rerun()

    st.write(f"Are you sure you want to exclude {variable}?")
    c1, c2 = st.columns(2)
    c1.button(
        "Yes",
        key="confirm-exclude-yes",
        help=f"Yes, exclude {variable} from the remainder of my analysis.",
        on_click=lambda: _on_click_yes(variable),
    )
    c2.button("No", key="confirm-exclude-no", on_click=lambda: st.rerun())


def exlude_variable_button(variable: str) -> None:
    """Exclude a variable."""
    st.button(
        "Exclude Variable",
        key="exclude-variable-button",
        on_click=lambda: confirm_excluded_variable(variable),
    )


def update_state(key: str, value: Any) -> None:  # noqa: ANN401
    """Update the state with a new key-value pair."""
    st.session_state.update(**{key: value}) if key in st.session_state else None


def get_data() -> pd.DataFrame:
    """Get the data."""
    return st.session_state["data"].drop(columns=st.session_state["excluded_variables"])


def get_target() -> pd.Series:
    """Get the target variable."""
    return get_data()[st.session_state["target_variable"]]


def get_excluded_features() -> list[str]:
    """Get the excluded variables."""
    return st.session_state["excluded_variables"]


def get_features() -> list[str]:
    """Get the features."""
    return [
        x
        for x in get_data().columns.tolist()
        if x
        not in [
            "fold",
            "folds",
            st.session_state["target_variable"],
            *get_excluded_features(),
        ]
    ]


def exclude_feature(feature: str) -> None:
    """Exclude a feature from the data."""
    st.session_state["excluded_variables"].append(
        feature
    ) if feature in get_data().columns else None


def is_data_loaded() -> bool:
    """Check if data is loaded."""
    return get_data().shape != (0, 0)


def two_column_layout_with_spacers() -> tuple:
    """Create a two-column layout with spacers."""
    _, col1, _, col2, _ = st.columns([0.05, 0.4, 0.1, 0.4, 0.05])
    return col1, col2
