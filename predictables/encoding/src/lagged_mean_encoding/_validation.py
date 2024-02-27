import polars as pl
from functools import wraps
from typing import Callable


def validate(func: Callable) -> Callable:
    """
    Decorator to validate that the first argument of a function is
    a LazyFrame, not a DataFrame (either pandas or polars).
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        lf = args[0]
        if not isinstance(lf, pl.LazyFrame):
            raise TypeError(f"Expected a LazyFrame, but got {type(lf)} instead.")
        return func(*args, **kwargs)

    return wrapper
