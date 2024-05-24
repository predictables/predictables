"""Define a data generator that produces the X and y data for the app."""

from __future__ import annotations
import streamlit as st
import polars as pl
from predictables.app.src.util import get_data


def X_train(
    data: pl.LazyFrame,
    target_variable: str,
    fold: int | None = None,
    is_time_series_validation: bool = True,
) -> pl.LazyFrame | None:
    """Return the X data for the app."""
    if fold is not None:
        data = (
            data.filter(pl.col("fold") < fold)
            if is_time_series_validation
            else data.filter(pl.col("fold") != fold)
        )
    return data.drop([target_variable, "fold"])


def y_train(
    data: pl.LazyFrame,
    target_variable: str,
    fold: int | None = None,
    is_time_series_validation: bool = True,
) -> pl.Series | None:
    """Return the y data for the app."""
    if fold is not None:
        data = (
            data.filter(pl.col("fold") < fold)
            if is_time_series_validation
            else data.filter(pl.col("fold") != fold)
        )
    return data.select(target_variable).collect().to_series()


def X_test(
    data: pl.LazyFrame,
    target_variable: str,
    fold: int | None = None,
    is_time_series_validation: bool = True,
) -> pl.LazyFrame | None:
    """Return the X data for the app."""
    if fold is not None:
        data = (
            data.filter(pl.col("fold") == fold)
            if is_time_series_validation
            else data.filter(pl.col("fold") == fold)
        )
    return data.drop([target_variable, "fold"])


def y_test(
    data: pl.LazyFrame,
    target_variable: str,
    fold: int | None = None,
    is_time_series_validation: bool = True,
) -> pl.Series | None:
    """Return the y data for the app."""
    if fold is not None:
        data = (
            data.filter(pl.col("fold") == fold)
            if is_time_series_validation
            else data.filter(pl.col("fold") == fold)
        )
    return data.select(target_variable).collect().to_series()


def X_y_gen(
    return_cv_folds: bool = True, is_time_series_validation: bool = True
) -> tuple:
    """Yield the X and y data for the app."""
    data = pl.from_pandas(get_data()).lazy()
    target_variable = st.session_state.target_variable

    params = {
        "data": data,
        "target_variable": target_variable,
        "is_time_series_validation": is_time_series_validation,
    }

    if return_cv_folds:
        for fold in (
            data.select("fold").unique().sort("fold").collect().to_pandas()["fold"]
        ):
            params["fold"] = fold
            yield (
                X_train(**params),
                y_train(**params),
                X_test(**params),
                y_test(**params),
            )

    else:
        yield X_train(**params), y_train(**params), X_test(**params), y_test(**params)