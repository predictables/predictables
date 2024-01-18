from typing import Union

import numpy as np
import pandas as pd
import polars as pl
from sklearn.linear_model import LogisticRegression

from PredicTables.util import to_pd_df, to_pd_s


def fit_sk_logistic_regression(
    X: Union[np.ndarray, pd.DataFrame, pl.DataFrame, pl.LazyFrame],
    y: Union[np.ndarray, pd.Series, pl.Series],
    fit_intercept: bool = False,
) -> LogisticRegression:
    """
    Fit a logistic regression model using the scikit-learn library. Used in the univariate
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
    sklearn.linear_model.LogisticRegression
        The fitted model
    """
    X = to_pd_df(X)
    y = to_pd_s(y).values.ravel()

    return LogisticRegression(fit_intercept=fit_intercept).fit(X, y)