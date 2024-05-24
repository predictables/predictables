"""Define Shap functionality for use in the app."""

from __future__ import annotations

import shap
import streamlit as st
import pandas as pd
from streamlit_shap import st_shap
from catboost import CatBoostClassifier


@st.cache_data
def shap_values(
    _model: CatBoostClassifier,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame | None = None,
) -> shap.Explanation:
    """Get the SHAP values for the model and the data."""
    explainer = shap.Explainer(_model, X_train)
    return explainer(X_test) if X_test is not None else explainer(X_train)


@st.cache_data
def shap_bar_plot(
    _shap_values: shap.Explanation,
    max_display: int = 10,
    cohorts: int = 2,
    height: int = 500,
    width: int = 700,
    use_clustering: bool = False,
    clustering_cutoff: float = 0.5,
    X: pd.DataFrame | None = None,
    y: pd.Series | None = None,
) -> None:
    """Display a bar plot of the SHAP values."""
    use_cohorts = cohorts > 1
    use_clustering = use_clustering and (X is not None) and (y is not None)

    if use_cohorts and use_clustering:
        st.error("Cannot use cohorts and clustering at the same time.")
        return
    elif use_cohorts:
        st_shap(
            shap.plots.bar(
                _shap_values.cohorts(cohorts).abs.mean(0), max_display=max_display
            ),
            height=height,
            width=width,
        )
    elif use_clustering:
        st_shap(
            shap.plots.bar(
                _shap_values.abs.mean(0),
                max_display=max_display,
                clustering=shap_clusters(X, y),
                clustering_cutoff=clustering_cutoff,
            ),
            height=height,
            width=width,
        )
    else:
        st_shap(
            shap.plots.bar(_shap_values.abs.mean(0), max_display=max_display),
            height=height,
            width=width,
        )


@st.cache_data
def shap_beeswarm_plot(
    _shap_values: shap.Explanation,
    max_display: int = 10,
    height: int = 500,
    width: int = 700,
    order_by: str = "mean",
    reverse_order: bool = False,
) -> None:
    """Display a beeswarm plot of the SHAP values."""
    reverse_factor = -1 if reverse_order else 1
    order_map = {
        "mean": _shap_values.abs.mean(0) * reverse_factor,
        "sum": _shap_values.abs.sum(0) * reverse_factor,
        "max": _shap_values.abs.max(0) * reverse_factor,
        "min": _shap_values.abs.min(0) * reverse_factor,
    }

    st_shap(
        shap.plots.beeswarm(
            _shap_values, max_display=max_display, order=order_map[order_by]
        ),
        height=height,
        width=width,
    )


@st.cache_data
def shap_clusters(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    return shap.utils.hclust(X, y)