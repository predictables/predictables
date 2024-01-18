import numpy as np


def harmonic_mean(*args: float) -> float:
    """
    Returns the harmonic mean of the given numbers.

    Parameters
    ----------
    *args : numeric
        Numbers to be averaged.

    Returns
    -------
    float
        The harmonic mean of the given numbers.
    """
    # if all arguments are zero, return zero
    if all(arg == 0 for arg in args):
        return 0

    denom = sum((1 / np.float64(arg)) for arg in args if arg != 0)

    # Return the harmonic mean
    if denom != 0:
        return len(args) / denom
    else:
        return 0
