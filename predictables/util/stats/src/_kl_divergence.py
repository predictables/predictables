from typing import Union

import numpy as np
import pandas as pd
import polars as pl

from predictables.util import to_pd_s


def kl_divergence(
    observed: Union[pd.Series, pl.Series, list, np.ndarray],
    modeled: Union[pd.Series, pl.Series, list, np.ndarray],
) -> float:
    """
    Calculate the KL divergence between two distributions. This function is
    a wrapper for scipy.stats.entropy.

    Parameters
    ----------
    observed : Union[pd.Series, pl.Series, list, np.ndarray]
        Observed values. Must be the same length as modeled values.
    modeled : Union[pd.Series, pl.Series, list, np.ndarray]
        Modeled values. Must be the same length as observed values.

    Returns
    -------
    float
        KL divergence between the two distributions
    """
    from scipy.stats import entropy  # type: ignore

    # Convert to pandas Series
    observed_ = to_pd_s(observed)
    modeled_ = to_pd_s(modeled)

    # Calculate KL divergence directly from observed and modeled averages
    return entropy(observed_, modeled_)
