from typing import List

import pandas as pd


def get_unique(x: pd.Series) -> List:
    """
    Returns a sorted list of the unique elements from the series `x`. My testing has
    shown this is either the most efficient method or within the very topmost efficent.

    Parameters
    ----------
    x : pd.Series
        A pandas series.

    Returns
    -------
    List
        A sorted list of the unique elements from the series `x`.
    """
    return list(sorted(set(x)))
