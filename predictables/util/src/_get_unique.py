from typing import List, Union

import pandas as pd
import polars as pl

from predictables.util import to_pd_s


def get_unique(x: Union[pd.Series, pl.Series]) -> List:
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
    return list(sorted(set(to_pd_s(x))))
