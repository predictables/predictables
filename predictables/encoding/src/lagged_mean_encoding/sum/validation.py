"""Validate the input arguments for the lagged mean encoding functions using decorators."""

from __future__ import annotations
from functools import wraps
import typing


def validate_offset(func: typing.Callable, arg_idx: int = -2) -> typing.Callable:
    """Validate that the offset is a non-negative integer."""

    @wraps(func)
    def wrapper(*args, **kwargs) -> typing.Callable:
        lag = args[arg_idx]
        if lag < 0:
            raise ValueError("Offset must be a non-negative integer")
        return func(*args, **kwargs)

    return wrapper


def validate_window(func: typing.Callable, arg_idx: int = -1) -> typing.Callable:
    """Validate that the window is a positive integer."""

    @wraps(func)
    def wrapper(*args, **kwargs) -> typing.Callable:
        window = args[arg_idx]
        if window < 0:
            raise ValueError("Window must be a positive integer")

        if window == 0:
            raise ValueError("Window must be a positive integer -- it cannot be zero")
        return func(*args, **kwargs)

    return wrapper


def validate_date_col(arg_idx: int, df_idx: int) -> typing.Callable:
    """Validate that the date column is a string."""

    def decorator(func: typing.Callable) -> typing.Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> typing.Callable:
            date_col = args[arg_idx]
            if not isinstance(date_col, str):
                raise ValueError("Date column must be a string")

            if date_col not in args[df_idx].columns:
                raise ValueError(
                    f"Date column ({date_col}) not found in DataFrame:\n{args[df_idx].columns}"
                )
            return func(*args, **kwargs)

        return wrapper

    return decorator