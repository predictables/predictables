from typing import Union

import numpy as np
import pandas as pd
import polars as pl
from sklearn.linear_model import LinearRegression  # type: ignore

from predictables.util import to_pd_df, to_pd_s


def fit_sk_linear_regression(
    X: Union[np.ndarray, pd.DataFrame, pl.DataFrame, pl.LazyFrame],
    y: Union[np.ndarray, pd.Series, pl.Series],
    fit_intercept: bool = False,
) -> LinearRegression:
    """
    Fit a linear regression model using the scikit-learn library. Used in the univariate
    analysis to fit a simple model to each variable.

    Parameters
    ----------
    X : Union[np.ndarray, pd.DataFrame, pl.DataFrame, pl.LazyFrame]
        The independent variables
    y : Union[np.ndarray, pd.Series, pl.Series]
        The dependent variable. Should be a 1D array.
    fit_intercept : bool, optional
        Whether to fit an intercept in the model, by default False. I think this is
        best for this use case, but it's not the default in sklearn.

    Returns
    -------
    sklearn.linear_model.LinearRegression
        The fitted model
    """
    # Validate X input
    if not (
        isinstance(X, np.ndarray)
        | isinstance(X, pd.DataFrame)
        | isinstance(X, pl.DataFrame)
        | isinstance(X, pl.LazyFrame)
    ):
        raise TypeError(
            "X must be one of np.ndarray, pd.DataFrame, pl.DataFrame, pl.LazyFrame. "
            f"Got {type(X)}"
        )

    # Validate y input
    if not (
        isinstance(y, np.ndarray) | isinstance(y, pd.Series) | isinstance(y, pl.Series)
    ):
        raise TypeError(
            f"y must be one of np.ndarray, pd.Series, pl.Series. Got {type(y)}"
        )

    # Validate fit_intercept input
    if not isinstance(fit_intercept, bool):
        raise TypeError(f"fit_intercept must be a bool. Got {type(fit_intercept)}")

    X_ = to_pd_df(X) if isinstance(X, (pl.DataFrame, pl.LazyFrame, pd.DataFrame)) else X
    y_ = to_pd_s(y).values.ravel() if isinstance(y, (pl.Series, pd.Series)) else y

    return LinearRegression(fit_intercept=fit_intercept).fit(X_, y_)
