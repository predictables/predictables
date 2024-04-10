"""Fit a time series model to the lagged mean-encoded categorical variable, and predict the current value of the mean-encoded variable.

Takes a dataframe with the lagged mean-encoded values, an id column, and the
target, and fits a time series model to the lagged mean-encoded values. The
model is then used to predict the current value of the mean-encoded variable.

Each observation in the dataframe is a time series of the mean-encoded
values of a categorical variable, with 18 observations per time series.
We use time-series cross-validation to fit the model using the most recent
12 observations to predict the 13th observation. The model is validated this
way 6 times, using the most recent 12 observations to predict the 13th, 14th,
15th, 16th, 17th, and 18th observations.

The 12 observations are a sliding window, so the first validation set uses
observations 1-12 to predict 13, the second validation set uses observations
2-13 to predict 14, and so on.
"""
from __future__ import annotations
import polars as pl
import lightgbm as lgb

from warnings import UserWarning

from predictables.util.transform import log_transform, logit_transform

class TimeSeriesEncoding:
    def __init__(self, lf: pl.LazyFrame, X_cols: list[str], y_col: str, id_column: str):
        """Initialize the TimeSeriesEncoding object.

        Parameters
        ----------
        lf : pl.LazyFrame
            The dataframe containing the lagged mean-encoded values.
        X_cols : list of str
            The names of the columns to use as features in the time series model.
        y_col : str
            The name of the target column.
        id_column : str
            The name of the column containing the time series id.
        """
        self._lf = lf
        self._X_cols = X_cols
        self._y_col = y_col
        self._id_column = id_column

        order_warning_msg = "The columns:\n{X_cols}\nare assumed to be pre-sorted most recent to least recent. If this is not the case, please sort the column names and recreate the `TimeSeriesEncoding` object."
        
        # Warn the user that the columns are assumed to be in order and continue
        if not self._lf[X_cols].is_sorted(by=[id_column]):
            UserWarning(order_warning_msg.format(X_cols=X_cols))

    def _is_sorted(self) -> bool:
        """Check if the columns are sorted by the `lag` value."""
        cols = [col.split("lag:")[1] for col in self._X_cols] 
        



    def get_X(self) -> pl.LazyFrame: