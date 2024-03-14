from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl

from predictables.util.src._to_pd import to_pd_s


def get_column_dtype(s: pl.Series | pd.Series | np.ndarray | list | tuple) -> str:
    """Return the dtype of a series as a string.

    The dtype is determined by checking the series against a set of rules.
    The rules are applied in the following order:

    1. If the series is numeric, check if it is binary, categorical,
       integer, or float.
    2. If the series is not numeric, check if it is a date.
    3. If the series is not numeric or a date, check if it binary, then
       categorical, then text, then boolean, then null.

    Parameters
    ----------
    s : Union[pl.Series, pd.Series, np.ndarray, list, tuple]
        The series to check.

    Returns
    -------
    str
        The dtype of the series as a string.
    """
    if is_numeric(s):
        if is_binary_integer(s):
            return "binary"
        elif is_categorical_integer(s):
            return "categorical"
        else:
            return "continuous"
    elif is_datetime(s):
        return "datetime"
    elif is_binary(s):
        return "binary"
    elif is_categorical(s):
        return "categorical"
    elif is_text(s):
        return "text"
    elif is_boolean(s):
        return "binary"
    elif is_null(s):
        return "null"
    else:
        raise TypeError("Unknown dtype")


def is_numeric(s: pl.Series | pd.Series | np.ndarray | list | tuple) -> bool:
    """Return True if the series is numeric, False otherwise.

    Parameters
    ----------
    s : Union[pl.Series, pd.Series, np.ndarray, list, tuple]
        The series to check.

    Returns
    -------
    bool
        True if the series is numeric, False otherwise.
    """
    try:
        to_pd_s(s).astype(str).astype(float)
        return True
    except (TypeError, ValueError):
        return False


def is_integer(s: pl.Series | pd.Series | np.ndarray | list | tuple) -> bool:
    """Return True if the series is integer, False otherwise.

    Parameters
    ----------
    s : Union[pl.Series, pd.Series, np.ndarray, list, tuple]
        The series to check.

    Returns
    -------
    bool
        True if the series is integer, False otherwise.
    """
    try:
        pd_series = to_pd_s(s)
        float_series = pd_series.astype(float)
        int_series = float_series.astype(int)
        return (float_series == int_series).all()
    except (TypeError, ValueError):
        return False


def is_binary_integer(s: pl.Series | pd.Series | np.ndarray | list | tuple) -> bool:
    """Return True if the series is binary integer, False otherwise.

    A series with integer values is considered to be binary if it has exactly two
    unique values. Having two unique values is assumed to be a sufficient
    condition for a binary integer series.

    Parameters
    ----------
    s : Union[pl.Series, pd.Series, np.ndarray, list, tuple]
        The series to check.

    Returns
    -------
    bool
        True if the series is an integer with only two unique values,
        False otherwise.
    """
    if is_integer(s):
        s = (
            to_pd_s(s)
            .astype(str)
            .astype(float)
            .astype(int)
            .drop_duplicates()
            .sort_values()
        )
        return s.shape[0] == 2  # noqa: PLR2004
    else:
        return False


def is_categorical_integer(
    s: pl.Series | pd.Series | np.ndarray | list | tuple,
) -> bool:
    """Return True if the series is an integer, False otherwise.

    A series is considered to be a categorical integer, if it has more than
    two unique numerical (integer) values, but where the difference between
    consecutive values is always exactly 1. When the difference between
    consecutive values is always 1, it is assumed that the integers are used
    to encode categories.

    Parameters
    ----------
    s : Union[pl.Series, pd.Series, np.ndarray, list, tuple]
        The series to check.

    Returns
    -------
    bool
        True if the series is an integer with more than two unique values, but
        where the difference between consecutive values is always 1, False otherwise.
    """
    if is_integer(s) and (not is_binary_integer(s)):
        s = (
            to_pd_s(s)
            .astype(str)
            .astype(float)
            .astype(int)
            .drop_duplicates()
            .sort_values()
        )
        all_diffs = s.diff().dropna()
        return all_diffs.min() == 1 and all_diffs.max() == 1
    else:
        return False


def is_datetime(s: pl.Series | pd.Series | np.ndarray | list | tuple) -> bool:
    """Return True if the series is a datetime, False otherwise.

    Parameters
    ----------
    s : Union[pl.Series, pd.Series, np.ndarray, list, tuple]
        The series to check.

    Returns
    -------
    bool
        True if the series is a datetime, False otherwise.
    """
    # Handle Polars Series
    if isinstance(s, pl.Series):
        if is_numeric(s) or is_integer(s):
            return False
        elif (
            (s.dtype == pl.datatypes.Datetime)
            or (s.dtype == pl.datatypes.Date)
            or (s.dtype == pl.datatypes.Time)
            or (s.dtype == pl.datatypes.Duration)
        ):
            return True
        elif (
            (s.dtype == pl.datatypes.Object)
            or (s.dtype == pl.datatypes.Utf8)
            or (s.dtype == pl.datatypes.Categorical)
        ):
            try:
                pd.to_datetime(to_pd_s(s)[0])
                return True
            except (ValueError, TypeError):
                return False
        else:
            return False

    # Handle Pandas Series
    try:
        if isinstance(to_pd_s(s).dtype, pd.DatetimeTZDtype) | isinstance(
            to_pd_s(s).dtype, np.dtypes.DateTime64DType
        ):
            return True
    except TypeError:
        pass

    # Check if the series is numeric or integer, and so is not a datetime
    if is_numeric(s) or is_integer(s):
        return False

    # Check if the series is string or categorical and can be parsed as dates
    try:
        _ = to_pd_s(s)
        if to_pd_s(s).dtype == "object" or isinstance(
            to_pd_s(s).dtype, pd.CategoricalDtype
        ):
            try:
                # Attempt to parse the first element to check if it's a date format
                pd.to_datetime(to_pd_s(s)[0])
                return True
            except (ValueError, TypeError):
                return False
    except TypeError:
        pass

    return False


def is_binary(s: pl.Series | pd.Series | np.ndarray | list | tuple) -> bool:
    """Return True if the series is binary, False otherwise.

    A series is considered binary if it has only two unique values.

    Parameters
    ----------
    s : Union[pl.Series, pd.Series, np.ndarray, list, tuple]
        The series to check.

    Returns
    -------
    bool
        True if the series is binary, False otherwise.
    """
    if is_numeric(s):
        return is_binary_integer(s)
    elif isinstance(s, pl.Series):
        return s.n_unique() == 2  # noqa: PLR2004
    elif isinstance(s, pd.Series):
        return s.nunique() == 2  # noqa: PLR2004
    elif isinstance(s, np.ndarray):
        return np.unique(s).shape[0] == 2  # noqa: PLR2004
    elif isinstance(s, (list, tuple)):
        return len(set(s)) == 2  # noqa: PLR2004
    else:
        try:
            try:
                out = to_pd_s(s).nunique() == 2  # noqa: PLR2004
            except Exception:
                out = pd.Series(s).nunique() == 2  # noqa: PLR2004

        except Exception:
            out = False
        return out


def is_categorical(s: pl.Series | pd.Series | np.ndarray | list | tuple) -> bool:
    """Return True if the series is categorical, False otherwise.

    A series is considered categorical if it has more than two
    unique values.

    Parameters
    ----------
    s : Union[pl.Series, pd.Series, np.ndarray, list, tuple]
        The series to check.

    Returns
    -------
    bool
        True if the series is categorical, False otherwise.
    """
    # Common sources of false positives: dates, binary integers
    if is_datetime(s):
        return False
    elif is_numeric(s):
        if is_binary_integer(s):
            return False

    if is_numeric(s):
        return is_categorical_integer(s)
    else:
        try:
            try:
                out = to_pd_s(s).nunique() > 2
            except Exception:
                out = pd.Series(s).nunique() > 2  # noqa: PLR2004
        except Exception:
            out = False
        return out


def is_text(s: pl.Series | pd.Series | np.ndarray | list | tuple) -> bool:
    """Return True if the series is text, False otherwise.

    A series is considered text if it has more than two unique values
    and is not categorical.

    Parameters
    ----------
    s : Union[pl.Series, pd.Series, np.ndarray, list, tuple]
        The series to check.

    Returns
    -------
    bool
        True if the series is text, False otherwise.
    """
    return (not is_categorical(s)) and (not is_binary(s)) and (not is_numeric(s))


def is_boolean(s: pl.Series | pd.Series | np.ndarray | list | tuple) -> bool:
    """Return True if the series is boolean, False otherwise.

    A series is considered boolean if it has two unique values and is not
    binary or categorical.

    Parameters
    ----------
    s : Union[pl.Series, pd.Series, np.ndarray, list, tuple]
        The series to check.

    Returns
    -------
    bool
        True if the series is boolean, False otherwise.
    """
    return (
        (not is_categorical(s))
        and (not is_binary(s))
        and (not is_numeric(s))
        and (to_pd_s(s).drop_duplicates().shape[0] == 2)
    )


def is_null(s: pl.Series | pd.Series | np.ndarray | list | tuple) -> bool:
    """Return True if the series is null, False otherwise.

    A series is considered null if it has only one unique value.

    Parameters
    ----------
    s : Union[pl.Series, pd.Series, np.ndarray, list, tuple]
        The series to check.

    Returns
    -------
    bool
        True if the series is null, False otherwise.
    """
    return to_pd_s(s).drop_duplicates().shape[0] == 1
