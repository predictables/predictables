from functools import wraps
from typing import Callable

import numpy as np
import pandas as pd
import polars as pl

from predictables.util.src._to_pl import to_pl_lf


def validate_lf(func: Callable) -> Callable:
    """Validate that the first argument of a function is a LazyFrame.

    Decorator to validate that the first argument of a function is
    a LazyFrame, not a DataFrame (either pandas or polars).
    """

    @wraps(func)
    def wrapper(*args, **kwargs):  # noqa: ANN202
        lf = kwargs.get("lf", args[0] if args else None)
        if isinstance(lf, (pd.DataFrame, pl.DataFrame, np.ndarray)):
            lf_ = to_pl_lf(lf)
        elif isinstance(lf, pl.LazyFrame):
            lf_ = lf
        else:
            raise ValueError(
                f"Expected a LazyFrame or DataFrame, got {type(lf)} instead."
            )

        if "lf" in kwargs:
            kwargs["lf"] = lf_
        else:
            args = (lf_,) + args[1:]

        return func(*args, **kwargs)

    return wrapper
