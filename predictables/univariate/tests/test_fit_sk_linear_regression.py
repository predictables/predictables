import numpy as np
import pandas as pd
import polars as pl
import pytest
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression

from predictables.univariate.src._fit_sk_linear_regression import (
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


# input validation tests
def test_input_validation_sk_linear_regression_coef():
    pd_df_X1, pd_series_y1 = pd_df_X(), pd_series_y()
    with pytest.raises(TypeError) as e:
        fit_sk_linear_regression(X=1, y=1)
    assert (
        str(e.value)
        == "X must be one of np.ndarray, pd.DataFrame, pl.DataFrame, pl.LazyFrame. Got <class 'int'>"
    ), f"Expected TypeError to be raised, but got {e.value}"

    # test one right one wrong
    with pytest.raises(TypeError) as e:
        fit_sk_linear_regression(X=pd_df_X1, y=1)
    assert (
        str(e.value)
        == "y must be one of np.ndarray, pd.Series, pl.Series. Got <class 'int'>"
    ), f"Expected TypeError to be raised, but got {e.value}"

    with pytest.raises(TypeError) as e:
        fit_sk_linear_regression(X=1, y=pd_series_y1)
    assert (
        str(e.value)
        == "X must be one of np.ndarray, pd.DataFrame, pl.DataFrame, pl.LazyFrame. Got <class 'int'>"
    ), f"Expected TypeError to be raised, but got {e.value}"

    # test fit_intercept
    with pytest.raises(TypeError) as e:
        fit_sk_linear_regression(X=pd_df_X1, y=pd_series_y1, fit_intercept=1)
    assert (
        str(e.value) == "fit_intercept must be a bool. Got <class 'int'>"
    ), f"Expected TypeError to be raised, but got {e.value}"
