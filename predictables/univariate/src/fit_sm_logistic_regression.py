from typing import Union

import numpy as np
import pandas as pd
import polars as pl
import statsmodels.api as sm

from predictables.util import to_pd_df, to_pd_s


def fit_sm_logistic_regression(
    X: Union[np.ndarray, pd.DataFrame, pl.DataFrame, pl.LazyFrame],
    y: Union[np.ndarray, pd.Series, pl.Series],
) -> sm.GLM:
    """
    Fit a logistic regression model using the statsmodels library. Used in the univariate
    analysis to fit a simple model to each variable.

    Parameters
    ----------
    X : Union[np.ndarray, pd.DataFrame, pl.DataFrame, pl.LazyFrame]
        The independent variables
    y : Union[np.ndarray, pd.Series, pl.Series]
        The dependent variable. Should be a 1D array.

    Returns
    -------
    sm.GLM
        The fitted model

    """
    X = to_pd_df(X)
    y = (
        to_pd_s(y)
        .astype(str)
        .str.replace("0", "0.01")
        .str.replace("1", "0.99")
        .str.replace("0.00.99", "0.01")
        .astype(float)
    )
    return sm.GLM(y, X, family=sm.families.Binomial()).fit()
