from __future__ import annotations


import pandas as pd
import polars as pl
import numpy as np
import statsmodels.api as sm

from predictables.util import DebugLogger, to_pd_df, to_pd_s

dbg = DebugLogger(working_file="_fit_sm_logistic_regression.py")


def fit_sm_logistic_regression(
    X: np.ndarray | pd.DataFrame | pl.DataFrame | pl.LazyFrame,
    y: np.ndarray | pd.Series | pl.Series,
) -> sm.GLM:
    """Fit a logistic regression model using the statsmodels library.

    Used in the univariate analysis to fit a simple model to each variable.

    Parameters
    ----------
    X : np.ndarray | pd.DataFrame | pl.DataFrame | pl.LazyFrame
        The independent variables
    y : np.ndarray | pd.Series | pl.Series
        The dependent variable. Should be a 1D array.

    Returns
    -------
    sm.GLM
        The fitted model

    """
    X_ = to_pd_df(X).reset_index(drop=True)
    y_ = to_pd_s(y).reset_index(drop=True).astype(float)
    return sm.GLM(y_, X_, family=sm.families.Binomial()).fit(disp=True)
