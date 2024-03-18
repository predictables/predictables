import numpy as np
import pandas as pd
import polars as pl
import pytest
import statsmodels.api as sm
from sklearn.datasets import fetch_california_housing

from predictables.univariate.src._fit_sm_linear_regression import (
    fit_sm_linear_regression,
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
    return pd_df_X().to_numpy()


def np_y():
    return pd_series_y().to_numpy().ravel()


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
def test_fit_sm_linear_regression_type(X, y):
    regression_result = fit_sm_linear_regression(X, y)
    assert isinstance(
        regression_result, sm.regression.linear_model.RegressionResultsWrapper
    ), f"Expected the result to be a sm.regression.linear_model.RegressionResultsWrapper, but got {type(regression_result)}"


@pytest.mark.parametrize(
    "X, y, is_np",
    [
        (pd_df_X(), pd_series_y(), False),
        (pd_df_X(), pl_y(), False),
        (pl_df_X(), pl_y(), False),
        (pl_lf_X(), pl_y(), False),
        (pl_df_X(), pd_series_y(), False),
        (pl_lf_X(), pd_series_y(), False),
        (np_X(), pd_series_y(), True),
        (np_X(), pl_y(), True),
        (pd_df_X(), np_y(), False),
        (pl_df_X(), np_y(), False),
        (pl_lf_X(), np_y(), False),
        (np_X(), np_y(), True),
    ],
)
def test_fit_sm_linear_regression_coef(X, y, is_np):
    regression_result = fit_sm_linear_regression(X, y)
    assert (
        np.round(regression_result.params.iloc[0], 3) == 0.512
    ), f"Expected the coefficient to be 0.512, but got {np.round(regression_result.params.iloc[0], 3)}"
