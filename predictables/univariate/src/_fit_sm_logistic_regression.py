from __future__ import annotations

import warnings
from typing import Union

import pandas as pd
import polars as pl
import statsmodels.api as sm  # type: ignore

from predictables.util import DebugLogger, to_pd_df, to_pd_s

dbg = DebugLogger(working_file="_fit_sm_logistic_regression.py")


def fit_sm_logistic_regression(
    X: Union[pd.DataFrame, pl.DataFrame, pl.LazyFrame], y: Union[pd.Series, pl.Series]
) -> sm.GLM:
    """
    Fit a logistic regression model using the statsmodels library. Used in the
    univariate analysis to fit a simple model to each variable.

    Parameters
    ----------
    X : Union[pd.DataFrame, pl.DataFrame, pl.LazyFrame]
        The independent variables
    y : Union[pd.Series, pl.Series]
        The dependent variable. Should be a 1D array.

    Returns
    -------
    sm.GLM
        The fitted model

    """
    dbg.msg("Entering fit_sm_logistic_regression function")
    X_ = to_pd_df(X).reset_index(drop=True)
    y_ = to_pd_s(y).reset_index(drop=True).astype(float)

    dbg.msg(f"X_=\n{X_},\n\ny_=\n{y_}")

    def log_warning(message, category, filename, lineno, file=None, line=None):
        detailed_msg = (
            f"Warning: {message}, Category: {category.__name__}, "
            f"File: {filename}, Line: {lineno}"
        )
        dbg.msg(detailed_msg)

    # Log warnings to the debug logger
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        warnings.showwarning = log_warning
        sm_model = sm.GLM(y_, X_, family=sm.families.Binomial()).fit(disp=True)

        # Check if there are any warnings and log them
        if w:
            for warning in w:
                log_warning(
                    warning.message, warning.category, warning.filename, warning.lineno
                )

    return sm_model
