from typing import Union

import numpy as np
import pandas as pd  # type: ignore
import polars as pl

from predictables.util.src._to_pd import to_pd_df, to_pd_s
from predictables.util.src._to_pl import to_pl_lf, to_pl_s
from predictables.util.src.logging._DebugLogger import DebugLogger

dbg = DebugLogger(working_file="_cv_filter.py")


def filter_df_by_cv_fold(
    df: Union[pd.DataFrame, pl.DataFrame, pl.LazyFrame],
    f: int,
    folds: Union[pd.Series, pl.Series, np.ndarray],
    time_series_validation: bool = True,
    train_test: str = "train",
    return_type: str = "pd",
) -> Union[pd.DataFrame, pl.LazyFrame]:
    """
    Filter a dataframe based on cross-validation fold. Returns a dataframe
    filtered for the given fold.

    Parameters
    ----------
    df : pd.DataFrame, pl.DataFrame, or pl.LazyFrame
        The dataframe to filter.
    f : int
        The fold to filter for.
    folds : pd.Series, pl.Series, or np.ndarray
        The column containing the fold information.
    time_series_validation : bool, default=True
        Whether the cross-validation is based on a time series.
    train_test : str, default="train"
        Whether to return the training or test set. If "train", returns the
        training set. If "test", returns the test set.
    return_type : str, default="pd"
        The type of dataframe to return. If "pd", returns a pandas dataframe. If "pl",
        returns a polars lazyframe.

    Returns
    -------
    pd.DataFrame or pl.LazyFrame
        The filtered dataframe.
    """
    if return_type.lower() in ["pd", "pandas", "np", "numpy", "pd.DataFrame"]:
        folds = to_pd_s(folds)
        df = to_pd_df(df)
        return to_pd_df(
            df.loc[cv_filter(f, folds, time_series_validation, train_test, "pd")]
        )

    elif return_type in ["pl", "polars", "pl.DataFrame"]:
        folds = to_pl_s(folds)
        df = to_pl_lf(df)
        return to_pl_lf(
            df.filter(cv_filter(f, folds, time_series_validation, train_test, "pl"))
        )
    else:
        dbg.msg(f"Unknown return type: {return_type}. Defaulting to pandas.")
        folds = to_pd_s(folds)
        df = to_pd_s(df)
        return to_pd_df(
            df.loc[cv_filter(f, folds, time_series_validation, train_test, "pd")]
        )


def filter_by_cv_fold(
    s: Union[pd.Series, pl.Series, np.ndarray],
    f: int,
    folds: Union[pd.Series, pl.Series, np.ndarray],
    time_series_validation: bool = True,
    train_test: str = "train",
    return_type: str = "pd",
) -> Union[pd.Series, pl.Series]:
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
    return_type : str, default="pd"
        The type of series to return. If "pd", returns a pandas series. If "pl",
        returns a polars series.

    Returns
    -------
    pd.Series or pl.Series
        The filtered series.
    """
    # make sure `folds` and `s` are pandas series
    if return_type.lower() in [
        "pd",
        "pandas",
        "np",
        "numpy",
        "pd.Series",
        "np.ndarray",
    ]:
        folds_ = to_pd_s(folds).reset_index(drop=True)
        s_ = to_pd_s(s).reset_index(drop=True)
        # filter based on cross-validation fold
        idx = cv_filter(f, folds_, time_series_validation, train_test, "pd").values
        print(f"\nidx.shape: {idx.shape}\n")
        return s_[idx]

    elif return_type in ["pl", "polars", "pl.Series"]:
        folds = to_pl_s(folds)
        s = to_pl_s(s)
        # filter based on cross-validation fold
        return s.filter(cv_filter(f, folds, time_series_validation, train_test, "pl"))
    else:
        dbg.msg(f"Unknown return type: {return_type}. Defaulting to pandas.")
        folds_ = to_pd_s(folds)
        s_ = to_pd_s(s)
        # filter based on cross-validation fold
        return s_[cv_filter(f, folds_, time_series_validation, train_test)]


def cv_filter(
    fold,
    fold_col: Union[pd.Series, pl.Series, np.ndarray],
    time_series_validation: bool = True,
    train_test: str = "train",
    return_type: str = "pd",
) -> Union[pd.Series, pl.Series]:
    """
    Filter data based on cross-validation fold.
    """
    return (
        _cv_filter_ts(fold, fold_col, train_test, return_type)
        if time_series_validation
        else _cv_filter_no_ts(fold, fold_col, train_test, return_type)
    )


def _cv_filter_no_ts(
    fold: int,
    fold_col: Union[pd.Series, pl.Series, np.ndarray],
    train_test: str,
    return_type: str,
) -> Union[pd.Series, pl.Series]:
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
    return_type : str
        The type of series to return. If "pd", returns a pandas series. If "pl",
        returns a polars series.

    Returns
    -------
    pd.Series
        A boolean series indicating whether the fold is equal to the given fold.
    """
    if return_type.lower() in [
        "pd",
        "pandas",
        "np",
        "numpy",
        "pd.Series",
        "np.ndarray",
    ]:
        pd_fold_col = to_pd_s(fold_col)
        return (
            pd_fold_col.ne(fold)
            if train_test.lower() == "train"
            else pd_fold_col.eq(fold)
        )
    elif return_type in ["pl", "polars", "pl.Series"]:
        pl_fold_col = to_pl_s(fold_col)
        return (
            pl_fold_col.ne(fold)
            if train_test.lower() == "train"
            else pl_fold_col.eq(fold)
        )
    else:
        dbg.msg(f"Unknown return type: {return_type}. Defaulting to pandas.")
        pd_fold_col = to_pd_s(fold_col)
        return (
            pd_fold_col.ne(fold)
            if train_test.lower() == "train"
            else pd_fold_col.eq(fold)
        )


def _cv_filter_ts(
    fold: int,
    fold_col: Union[pd.Series, pl.Series, np.ndarray],
    train_test: str,
    return_type: str,
) -> Union[pd.Series, pl.Series]:
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
    return_type : str
        The type of series to return. If "pd", returns a pandas series. If "pl",
        returns a polars series.

    Returns
    -------
    pd.Series or pl.Series
        A boolean series indicating whether the fold is less than or equal to the
        given fold.

    Notes
    -----
    When combining training and test sets, test "cv folds" do not exist, so are filled
    in with negative labels sometimes. This is why we use less than or equal to, but not
    negative.
    """
    return (
        _cv_filter_ts_pl(fold, fold_col, train_test)
        if return_type.lower() in ["pl", "polars", "pl.Series"]
        else _cv_filter_ts_pd(fold, fold_col, train_test)
    )


def _cv_filter_ts_pd(
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
    pd_fold_col = to_pd_s(fold_col)
    return_val_train = (pd_fold_col.lt(fold) & (pd_fold_col.ge(0))).reset_index(
        drop=True
    )
    return_val_test = pd_fold_col.eq(fold).reset_index(drop=True)

    return return_val_train if train_test.lower() == "train" else return_val_test


def _cv_filter_ts_pl(
    fold: int, fold_col: Union[pd.Series, pl.Series, np.ndarray], train_test: str
) -> pl.Series:
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
    pl.Series
        A boolean series indicating whether the fold is less than or equal to the
        given fold.

    Notes
    -----
    When combining training and test sets, test "cv folds" do not exist, so are filled
    in with negative labels sometimes. This is why we use less than or equal to, but not
    negative.
    """
    pd_fold_col = to_pd_s(fold_col)
    return_val_train = to_pl_s(
        ((pd_fold_col.lt(fold)) & (pd_fold_col.ge(0))).reset_index(drop=True)
    )
    return_val_test = to_pl_s(pd_fold_col.eq(fold).reset_index(drop=True))

    return return_val_train if train_test.lower() == "train" else return_val_test
