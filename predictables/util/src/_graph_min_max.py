from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd  # type: ignore
import polars as pl

from predictables.util.src._to_pd import to_pd_s


def graph_min_max(
    x: Union[pd.Series, pl.Series, np.ndarray],
    min_: Optional[float],
    max_: Optional[float],
) -> Tuple[float, float]:
    """
    This function is used to calculate the minimum and maximum values to plot on the x-axis of the density plot.

    Parameters
    ----------
    x : Union[pd.Series, pl.Series, np.ndarray]
        The variable to plot the density of.
    min_ : float, optional
        The minimum value to plot. If None, defaults to the minimum of x. Used
        to extend the curve to the edges of the plot.
    max_ : float, optional
        The maximum value to plot. If None, defaults to the maximum of x. Used
        to extend the curve to the edges of the plot.

    Returns
    -------
    Tuple[float, float]
        A tuple containing the minimum and maximum values to plot on the x-axis of the density plot.
    """
    x_ = to_pd_s(x)
    # If min and max are not provided, use the min and max of the data
    _min = x_.min() if min_ is None else min_
    _max = x_.max() if max_ is None else max_

    if _min > _max:
        raise ValueError(f"min_ ({_min}) should be less than or equal to max_ ({_max})")
    return _min, _max
