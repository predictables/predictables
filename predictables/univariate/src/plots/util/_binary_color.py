import pandas as pd


def binary_color(x: int) -> pd.Series:
    """
    Return a color for each value in x, based on whether it is 0 or 1.

    Parameters
    ----------
    x : int
        The value to get the color for.

    Returns
    -------
    int
        The color for x.
    """
    if x == 0:
        return "blue"
    elif x == 1:
        return "orange"
    else:
        raise ValueError(f"Invalid value {x} for binary variable.")
