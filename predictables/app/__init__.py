import pandas as pd
import streamlit as st
from typing import Any
import os
from dotenv import load_dotenv, find_dotenv
from sklearn.linear_model import LogisticRegressionCV
from .plot_src import histogram, scatter, boxplot
from .src import train_test_split

load_dotenv(find_dotenv())


def update_state(key: str, value: Any) -> None:  # noqa: ANN401
    """Update the state with a new key-value pair."""
    st.session_state.update(**{key: value}) if key in st.session_state else None


def initialize_state() -> None:
    """Initialize state variables if needed."""
    if "data" not in st.session_state:
        st.session_state["data"] = pd.DataFrame()

    if "columns" not in st.session_state:
        st.session_state["columns"] = []

    if "target_variable" not in st.session_state:
        st.session_state["target_variable"] = (
            "evolve_hit_count"
            if os.environ.get("TARGET_VARIABLE") is None
            else os.environ.get("TARGET_VARIABLE")
        )

    if "univariate_feature_variable" not in st.session_state:
        st.session_state["univariate_feature_variable"] = ""

    if "models" not in st.session_state:
        st.session_state["models"] = {}


def is_data_loaded() -> bool:
    """Check if data is loaded."""
    return st.session_state["data"].shape != (0, 0)

def two_column_layout_with_spacers() -> tuple:
    """Create a two-column layout with spacers."""
    _, col1, _, col2, _ = st.columns([0.05, 0.4, 0.1, 0.4, 0.05])
    return col1, col2

# @st.cache_data
# def build_models(data: pd.DataFrame, ts_cross_validation: bool = True) -> None:
#     """Build models using the data."""
#     models = {}
#     unique_folds = data["fold"].unique()

#     for fold in unique_folds:
#         if fold > 0:
#             qry = f"fold != {fold}" if not ts_cross_validation else f"fold <= {fold}"
#             filtered_data = data.query(qry)
#             X = filtered_data.drop(columns=["target"])
#             y = filtered_data["target"]

#             # Build a logistic regression model
#             model = LogisticRegressionCV(cv=5, max_iter=1000)
#             model.fit(X, y)

#             models[fold] = model

#     st.session_state.update(models=models)