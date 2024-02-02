from typing import Any, Union

import numpy as np
import pandas as pd
import polars as pl
import statsmodels.api as sm  # type: ignore

from predictables.util import to_pd_df, to_pd_s


def fit_sm_logistic_regression(
    X: Union[pd.DataFrame, pl.DataFrame, pl.LazyFrame],
    y: Union[pd.Series, pl.Series],
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
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X)
    else:
        X = to_pd_df(X)

    if isinstance(y, np.ndarray):
        y = pd.Series(y)
    else:
        y = to_pd_s(y)

    Y: Any = (
        y.astype(str)
        .str.replace("0", "0.01")
        .str.replace("1", "0.99")
        .str.replace("0.00.99", "0.01")
        .astype(float)
    )
    return sm.GLM(Y.astype(float), X, family=sm.families.Binomial()).fit()
