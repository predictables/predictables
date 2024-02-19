from typing import Union

import numpy as np
import pandas as pd
import polars as pl

from predictables.util.src._to_pd import to_pd_s


def cv_filter(
    fold,
    fold_col: Union[pd.Series, pl.Series, np.ndarray],
    time_series_validation: bool = False,
):
    """
    Filter data based on cross-validation fold.
    """
    print(f"ts valid: 7: {time_series_validation}")
    return (
        _cv_filter_ts(fold, fold_col)
        if time_series_validation
        else _cv_filter_no_ts(fold, fold_col)
    )


def _cv_filter_no_ts(
    fold: int, fold_col: Union[pd.Series, pl.Series, np.ndarray]
) -> pd.Series:
    """
    Filter data based on cross-validation fold. This is a helper function
    for cv_filter, and is used when the cross-validation is not based on a
    time series.

    Parameters
    ----------
    fold : int
        The fold to filter for.
    fold_col : pd.Series, pl.Series, or np.ndarray
        The column containing the fold information.

    Returns
    -------
    pd.Series
        A boolean series indicating whether the fold is equal to the given fold.
    """
    return to_pd_s(fold_col).ne(fold)


def _cv_filter_ts(
    fold: int, fold_col: Union[pd.Series, pl.Series, np.ndarray]
) -> pd.Series:
    """
    Fiter data based on cross-validation fold. This is a helper function for
    cv_filter, and is used when the cross-validation is based on a time series. In
    the case of a time series, you filter the data based on whether the fold is
    less than or equal to the given fold label -- fold labels are assigned to the
    data based on a date or time column.

    Parameters
    ----------
    fold : int
        The fold to filter for.
    fold_col : pd.Series, pl.Series, or np.ndarray
        The column containing the fold information.

    Returns
    -------
    pd.Series
        A boolean series indicating whether the fold is less than or equal to the
        given fold.

    Notes
    -----
    When combining training and test sets, test "cv folds" do not exist, so are filled
    in with negative labels sometimes. This is why we use less than or equal to, but not
    negative.
    """
    fold_col = pd.Series(fold_col)
    if isinstance(fold, pd.Series):
        fold = fold.iloc[0]  # or any other logic to extract the single value
    fold = int(fold)
    return_val = (fold_col < fold) & (fold_col >= 0)
    return return_val.reset_index(drop=True)
