"""Fit the model used in the app."""

from __future__ import annotations

import pandas as pd
import polars as pl
import polars.selectors as cs
import streamlit as st
from catboost import CatBoostClassifier

from predictables.app.src.data import X_y_gen


@st.cache_resource
def fit_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    **kwargs,
) -> CatBoostClassifier:
    """Fit the model used in the app."""
    cat_cols = (
        pl.from_pandas(X_train).lazy().select(cs.categorical()).columns
        + pl.from_pandas(X_train).lazy().select(cs.string()).columns
    )
    model = CatBoostClassifier(cat_features=cat_cols)
    model.fit(X_train, y_train, eval_set=(X_test, y_test), **kwargs)
    return model


@st.cache_resource
def fit_model_with_cross_validation(
    return_cv_folds: bool = True, is_time_series_validation: bool = True, **kwargs
) -> list[CatBoostClassifier]:
    """Fit the model used in the app with cross validation."""
    models = []
    for X_train, y_train, X_test, y_test in X_y_gen(
        return_cv_folds=return_cv_folds,
        is_time_series_validation=is_time_series_validation,
    ):
        models.append(fit_model(X_train, y_train, X_test, y_test, **kwargs))

    return models