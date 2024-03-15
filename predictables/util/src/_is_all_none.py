from __future__ import annotations

from typing import Union

import numpy as np
import pandas as pd
import polars as pl

from ._to_pd import to_pd_s
from ._to_pl import to_pl_s


def _is_all_none_pd(s: pd.Series) -> bool:
    """Check if a pandas series is all None."""
    return to_pd_s(s).isna().all()


def _is_all_none_pl(s: pl.Series) -> bool:
    """Check if a polars series is all None."""
    return to_pl_s(s).is_null().all()


def _is_all_none_np(s: np.ndarray) -> bool:
    """Check if a numpy array is all None."""
    if not isinstance(s, np.ndarray):
        raise TypeError("s must be a numpy array.")
    return np.isnan(s.astype(float)).all()


def is_all_none(s: Union[pd.Series, pl.Series]) -> bool:
    """Check if a series is all None."""
    if isinstance(s, pd.Series):
        return _is_all_none_pd(s)
    elif isinstance(s, pl.Series):
        return _is_all_none_pl(s)
    elif isinstance(s, np.ndarray):
        return _is_all_none_np(s)
    else:
        raise TypeError("s must be a pandas or polars series.")
