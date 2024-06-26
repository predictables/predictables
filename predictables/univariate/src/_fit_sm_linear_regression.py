from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
import statsmodels.api as sm

from predictables.util import to_pd_df, to_pd_s


def fit_sm_linear_regression(
    X: np.ndarray | pd.DataFrame | pl.DataFrame | pl.LazyFrame,
    y: np.ndarray | pd.Series | pl.Series,
) -> sm.regression.linear_model.RegressionResultsWrapper:
    """Fit a linear regression model using the statsmodels library.

    Used in the univariate analysis to fit a simple model to each variable.

    Parameters
    ----------
    X : np.ndarray | pd.DataFrame | pl.DataFrame | pl.LazyFrame
        The independent variables
    y : np.ndarray | pd.Series | pl.Series
        The dependent variable. Should be a 1D array.

    Returns
    -------
    sm.regression.linear_model.RegressionResultsWrapper
        The fitted model

    """
    X = to_pd_df(X).reset_index(drop=True)
    y = to_pd_s(y).reset_index(drop=True).to_numpy().ravel()
    return sm.OLS(y, X).fit()
