from typing import Union

import numpy as np
import pandas as pd
import polars as pl


def harmonic_mean(*args) -> float:
    """
    Returns the harmonic mean of the given, numbers, list, numpy array,
    pandas series, or polars series.

    Parameters
    ----------
    *args : numeric or numpy array
        Numbers or numpy array to be averaged.


    Returns
    -------
    float
        The harmonic mean of the given numbers or numpy array.
    """
    if isinstance(args[0], (int, float)):
        return _harmonic_mean_args(*args) if len(args) > 1 else args[0]
    elif isinstance(args[0], list):
        return _harmonic_from_list(args[0])
    elif isinstance(args[0], np.ndarray):
        return _harmonic_from_series(*args)
    elif isinstance(args[0], pd.Series):
        return _harmonic_from_series(args[0])
    elif isinstance(args[0], pl.Series):
        return _harmonic_from_series(args[0])
    else:
        raise TypeError(
            "Input must be a number, list, numpy array, pandas series, or polars "
            f"series. Got {type(args[0])} instead."
        )


def _harmonic_mean_args(*args) -> float:
    """
    Returns the harmonic mean of the given numbers or numpy array.

    Parameters
    ----------
    *args : numeric or numpy array
        Numbers or numpy array to be averaged.

    Returns
    -------
    float
        The harmonic mean of the given numbers or numpy array.
    """
    # If this is a single integer or float value, return that value (as a float)
    if len(args) == 1 and isinstance(args[0], (int, float)):
        return float(args[0])

    # Convert input to numpy array if it's not already
    elif isinstance(args[0], list):
        args_ = np.array(*args[0])
    elif isinstance(args[0], np.ndarray):
        args_ = args[0]
    elif len(args) > 1:
        args_ = np.array(args)
    else:
        raise TypeError(
            "Input must be a number, list, or numpy array. Got "
            f"{type(args[0])} instead."
        )

    # if all elements are zero, return zero
    if np.all(args_ == 0):
        return 0

    denom = 1 / args_[0] if len(args_) == 1 else np.sum(1 / args_[args_ != 0])

    # Return the harmonic mean
    return len(args_) / denom if denom != 0 else 0


def _harmonic_from_list(input_list):
    """
    Calculates the harmonic mean of a list or numpy array of inputs.

    Parameters
    ----------
    input_list : list or numpy array of float
        List or numpy array of numbers to be averaged.

    Returns
    -------
    float
        The harmonic mean of the given list or numpy array of inputs.
    """
    # Convert input to numpy array if it's not already
    if not isinstance(input_list, np.ndarray):
        input_list = np.array(input_list)

    # if all elements in the list are zero, return zero
    if np.all(input_list == 0):
        return 0

    denom = np.sum(1 / input_list[input_list != 0])

    # Return the harmonic mean
    return len(input_list) / denom if denom != 0 else 0


def _harmonic_from_series(s: Union[pd.Series, pl.Series, np.ndarray]):
    """
    Calculates the harmonic mean of a pandas, polars, or numpy series of inputs.

    Parameters
    ----------
    s : pandas, polars, or numpy series of float
        Series of numbers to be averaged.

    Returns
    -------
    float
        The harmonic mean of the given series of inputs.
    """
    if isinstance(s, (pd.Series, pl.Series)):
        s = s.to_numpy()
    elif not isinstance(s, np.ndarray):
        raise TypeError("Input must be a pandas, polars, or numpy series of numbers.")

    # if all elements in the series are zero, return zero
    if np.all(s == 0):
        return 0

    denom = np.sum(1 / s[s != 0])

    # Return the harmonic mean
    return len(s) / denom if denom != 0 else 0
