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
    if len(args) > 1:
        return _harmonic_mean_args(*args)
    elif isinstance(args[0], int) or isinstance(args[0], float):
        return args[0]
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
            f"Input must be a number, list, numpy array, pandas series, or polars series. Got {type(args[0])} instead."
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
    if len(args) == 1 and (isinstance(args, int) or isinstance(args, float)):
        return float(args)

    # Convert input to numpy array if it's not already
    if not isinstance(args[0], np.ndarray):
        args = np.array(args)

    # if all elements are zero, return zero
    if np.all(args == 0):
        return 0

    if len(args) == 1:
        denom = 1 / args[0]
    else:
        denom = np.sum(1 / args[args != 0])

    # Return the harmonic mean
    if denom != 0:
        return len(args) / denom
    else:
        return 0


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
    if denom != 0:
        return len(input_list) / denom
    else:
        return 0


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
    if isinstance(s, pd.Series):
        s = s.values
    elif isinstance(s, pl.Series):
        s = s.to_numpy()
    elif not isinstance(s, np.ndarray):
        raise TypeError("Input must be a pandas, polars, or numpy series of numbers.")

    # if all elements in the series are zero, return zero
    if np.all(s == 0):
        return 0

    denom = np.sum(1 / s[s != 0])

    # Return the harmonic mean
    if denom != 0:
        return len(s) / denom
    else:
        return 0
