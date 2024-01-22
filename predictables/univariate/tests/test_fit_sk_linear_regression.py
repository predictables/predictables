import numpy as np
import pandas as pd
import polars as pl
import pytest
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression

from predictables.univariate.src.fit_sk_linear_regression import (
    fit_sk_linear_regression,
)

california_housing = fetch_california_housing()
X, y = (
    pd.DataFrame(
        california_housing.data, columns=california_housing.feature_names
    ).iloc[:, :1],
    pd.Series(california_housing.target, name="target"),
)


def pd_df_X():
    return X


def pd_series_y():
    return y


def pl_df_X():
    return pl.from_pandas(pd_df_X())


def pl_lf_X():
    return pl.from_pandas(pd_df_X()).lazy()


def pl_y():
    return pl.from_pandas(pd_series_y())


def np_X():
    return pd_df_X().values


def np_y():
    return pd_series_y().values.ravel()


@pytest.mark.parametrize(
    "X, y",
    [
        (pd_df_X(), pd_series_y()),
        (pd_df_X(), pl_y()),
        (pl_df_X(), pl_y()),
        (pl_lf_X(), pl_y()),
        (pl_df_X(), pd_series_y()),
        (pl_lf_X(), pd_series_y()),
        (np_X(), pd_series_y()),
        (np_X(), pl_y()),
        (pd_df_X(), np_y()),
        (pl_df_X(), np_y()),
        (pl_lf_X(), np_y()),
        (np_X(), np_y()),
    ],
)
def test_fit_sk_linear_regression_type(X, y):
    regression_result = fit_sk_linear_regression(X, y)
    assert isinstance(
        regression_result, LinearRegression
    ), f"Expected the result to be a fitted sklearn.linear_model.LinearRegression object, but got {type(regression_result)}"


@pytest.mark.parametrize(
    "X, y",
    [
        (pd_df_X(), pd_series_y()),
        (pd_df_X(), pl_y()),
        (pl_df_X(), pl_y()),
        (pl_lf_X(), pl_y()),
        (pl_df_X(), pd_series_y()),
        (pl_lf_X(), pd_series_y()),
        (np_X(), pd_series_y()),
        (np_X(), pl_y()),
        (pd_df_X(), np_y()),
        (pl_df_X(), np_y()),
        (pl_lf_X(), np_y()),
        (np_X(), np_y()),
    ],
)
def test_fit_sk_linear_regression_coef(X, y):
    regression_result = fit_sk_linear_regression(X, y)
    assert (
        np.round(regression_result.coef_[0], 3) == 0.512
    ), f"Expected the coefficient to be 0.512, but got {np.round(regression_result.coef_[0], 3)}"
