from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
from sklearn.linear_model import LogisticRegression

from predictables.util import to_pd_df, to_pd_s


def fit_sk_logistic_regression(
    X: np.ndarray | pd.DataFrame | pl.DataFrame | pl.LazyFrame,
    y: np.ndarray | pd.Series | pl.Series,
    fit_intercept: bool = False,
) -> LogisticRegression:
    """Fit a logistic regression model using the scikit-learn library.

    Used in the univariate analysis to fit a simple model to each variable.

    Parameters
    ----------
    X : np.ndarray | pd.DataFrame | pl.DataFrame | pl.LazyFrame
        The independent variables
    y : np.ndarray | pd.Series | pl.Series
        The dependent variable. Should be a 1D array.
    fit_intercept : bool, optional
        Whether to fit an intercept in the model, by default False. I think this is
        best for this use case, but it's not the default in sklearn.

    Returns
    -------
    sklearn.linear_model.LogisticRegression
        The fitted model
    """
    X_ = (
        to_pd_df(X)
        if isinstance(X, (pd.DataFrame, pl.DataFrame, pl.LazyFrame))
        else pd.DataFrame(X)
    )
    y_ = (
        to_pd_s(y).reset_index(drop=True).to_numpy().ravel()
        if isinstance(y, (pd.Series, pl.Series))
        else pd.Series(y)
    )

    return LogisticRegression(fit_intercept=fit_intercept).fit(
        X_.reset_index(drop=True), y_
    )
