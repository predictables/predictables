from __future__ import annotations

import pandas as pd
import streamlit as st
from typing import Any
import os
from dotenv import load_dotenv, find_dotenv
from sklearn.linear_model import LogisticRegressionCV
from .plot_src import histogram, scatter, boxplot
from .src import (
    train_test_split,
    X_y_gen,
    fit_model,
    fit_model_with_cross_validation,
    pca,
    exlude_variable_button,
    confirm_excluded_variable,
)
from .src.shap import shap_values, shap_bar_plot, shap_beeswarm_plot

load_dotenv(find_dotenv())





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

    if "is_time_series_data" not in st.session_state:
        st.session_state["is_time_series_data"] = True

    if "excluded_variables" not in st.session_state:
        st.session_state["excluded_variables"] = []

