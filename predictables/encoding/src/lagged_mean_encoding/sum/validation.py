"""Validate the input arguments for the lagged mean encoding functions using decorators."""

from __future__ import annotations
from functools import wraps
import typing
import pandas as pd
import polars as pl


def validate_offset(func: typing.Callable) -> typing.Callable:
    """Validate that the offset is a non-negative integer."""

    @wraps(func)
    def wrapper(*args, **kwargs) -> typing.Callable:
        offset = (
            kwargs.get("offset")
            if "offset" in kwargs
            else args[-2]
            if len(args) > 1
            else 0
        )
        if not isinstance(offset, int) or offset < 0:
            raise ValueError(f"Offset must be a non-negative integer, got {offset}")
        return func(*args, **kwargs)

    return wrapper


def validate_window(func: typing.Callable) -> typing.Callable:
    """Validate that the window is a positive integer."""

    @wraps(func)
    def wrapper(*args, **kwargs) -> typing.Callable:
        window = (
            kwargs.get("window")
            if "window" in kwargs
            else args[-1]
            if len(args) > 1
            else 30
        )
        if not isinstance(window, int) or window <= 0:
            raise ValueError(f"Window must be a positive integer, got {window}")
        return func(*args, **kwargs)

    return wrapper


def validate_date_col(date_col_idx: int, df_idx: int) -> typing.Callable:
    """Validate that the date column is a string."""

    def decorator(func: typing.Callable) -> typing.Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> typing.Callable:
            date_col = (
                args[date_col_idx]
                if date_col_idx < len(args)
                else kwargs.get("date_col")
            )
            df = args[df_idx] if df_idx < len(args) else kwargs.get("df")
            if not isinstance(date_col, str):
                raise ValueError(
                    f"Date column must be a string, got {type(date_col).__name__}"
                )
            if not isinstance(df, pd.DataFrame | pl.DataFrame | pl.LazyFrame):
                raise ValueError(
                    f"The argument must be a pandas/polars DataFrame or LazyFrame. Got {type(df)}."
                )
            if date_col not in df.columns:
                raise ValueError(
                    f"Date column ({date_col}) not found in DataFrame columns: {df.columns}"
                )
            return func(*args, **kwargs)

        return wrapper

    return decorator

