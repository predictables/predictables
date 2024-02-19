from typing import Union

import numpy as np
import pandas as pd
import polars as pl

from predictables.util.src._to_pd import to_pd_s


def filter_by_cv_fold(
    s: Union[pd.Series, pl.Series, np.ndarray],
    f: int,
    folds: Union[pd.Series, pl.Series, np.ndarray],
    time_series_validation: bool = True,
    train_test: str = "train",
) -> pd.Series:
    """
    Filter data based on cross-validation fold. Returns a series filtered
    for the given fold.

    Parameters
    ----------
    s : pd.Series, pl.Series, or np.ndarray
        The column to filter.
    f : int
        The fold to filter for.
    folds : pd.Series, pl.Series, or np.ndarray
        The column containing the fold information.
    time_series_validation : bool, default=True
        Whether the cross-validation is based on a time series.
    train_test : str, default="train"
        Whether to return the training or test set. If "train", returns the
        training set. If "test", returns the test set.

    Returns
    -------
    pd.Series
        The filtered series.
    """
    # make sure `folds` and `s` are pandas series
    folds = to_pd_s(folds)
    s = to_pd_s(s)

    # filter based on cross-validation fold
    return s[cv_filter(f, folds, time_series_validation, train_test)]


def cv_filter(
    fold,
    fold_col: Union[pd.Series, pl.Series, np.ndarray],
    time_series_validation: bool = True,
    train_test: str = "train",
) -> pd.Series:
    """
    Filter data based on cross-validation fold.
    """
    return (
        _cv_filter_ts(fold, fold_col, train_test)
        if time_series_validation
        else _cv_filter_no_ts(fold, fold_col, train_test)
    )


def _cv_filter_no_ts(
    fold: int, fold_col: Union[pd.Series, pl.Series, np.ndarray], train_test: str
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
    train_test : str
        Whether to return the training or test set. If "train", returns the
        training set. If "test", returns the test set.

    Returns
    -------
    pd.Series
        A boolean series indicating whether the fold is equal to the given fold.
    """
    return (
        to_pd_s(fold_col).ne(fold)
        if train_test.lower() == "train"
        else to_pd_s(fold_col).eq(fold)
    )


def _cv_filter_ts(
    fold: int, fold_col: Union[pd.Series, pl.Series, np.ndarray], train_test: str
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
    train_test : str
        Whether to return the training or test set. If "train", returns the
        training set. If "test", returns the test set.

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
        fold = fold.iloc[0]
    fold = int(fold)
    return_val_train = ((fold_col < fold) & (fold_col >= 0)).reset_index(drop=True)
    return_val_test = (fold_col == fold).reset_index(drop=True)
    return return_val_train if train_test.lower() == "train" else return_val_test
