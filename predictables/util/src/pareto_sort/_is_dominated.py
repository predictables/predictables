from __future__ import annotations

from typing import Union

import numpy as np
import pandas as pd


def is_dominated(
    a: Union[np.ndarray, pd.Series], b: Union[np.ndarray, pd.Series]
) -> bool:
    """
    Check if a is dominated by b.

    Parameters
    ----------
    a : np.array
        An array representing a variable with its objectives.
    b : np.array
        An array representing another variable with its objectives.

    Returns
    -------
    bool
        True if a is dominated by b, False otherwise.
    """
    return np.logical_and(np.all(a <= b), np.any(a < b))
