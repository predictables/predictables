from __future__ import annotations

import typing
from functools import wraps

import pandas as pd
import polars as pl

from predictables.util.src._to_pl import to_pl_lf


def validate_column(
    func: typing.Callable,
    data_frame: pd.DataFrame | pl.DataFrame | pl.LazyFrame,
    column_name: str | None = None,
) -> typing.Callable:
    """Validate that a passed name represents a column in a DataFrame."""

    @wraps(func)
    def wrapper(*args, **kwargs) -> typing.Callable:
        col_name = _get_test_column_name(column_name, *args, **kwargs)
        df_ = _handle_df_types(data_frame)
        cols = _get_df_columns(df_)
        is_col_in_df, msg = _test_column_name_in_columns(col_name, cols)
        if is_col_in_df:
            return func(*args, **kwargs)

        # If the column is not found, raise an error
        raise ValueError(msg)

    return wrapper


def _get_test_column_name(column_name: str | None = None, *args, **kwargs) -> str:
    """Get the column name from the function arguments."""
    is_col_found = False
    if (len(args) == 0) and ("column" not in kwargs) and (column_name not in kwargs):
        raise ValueError("No arguments passed to function.")

    if column_name is not None:
        col_name = column_name
        is_col_found = True
    else:
        for c in [
            "column",
            "col",
            "date_col",
            "date_column",
            "x_col",
            "x_column",
            "index_col",
            "index_column",
        ]:
            if c in kwargs:
                col_name = c
                is_col_found = True
                break

    if not is_col_found:
        raise ValueError("No column name found in function arguments.")

    return col_name


def _handle_df_types(df: pd.DataFrame | pl.DataFrame | pl.LazyFrame) -> pl.LazyFrame:
    """Handle different DataFrame types."""
    if isinstance(df, pd.DataFrame):
        df_ = to_pl_lf(df)
    elif isinstance(df, pl.DataFrame):
        df_ = df.lazy()
    elif isinstance(df, pl.LazyFrame):
        df_ = df
    else:
        raise ValueError(f"Expected a DataFrame or LazyFrame, got {type(df)} instead.")
    return df_


def _get_df_columns(df: pd.DataFrame | pl.DataFrame | pl.LazyFrame) -> list[str]:
    """Return the column names of a DataFrame."""
    if isinstance(df, (pd.DataFrame, pl.DataFrame, pl.LazyFrame)):
        return df.columns.tolist() if isinstance(df, pd.DataFrame) else df.columns

    raise ValueError(f"Expected a DataFrame or LazyFrame, got {type(df)} instead.")


def _test_column_name_in_columns(
    col_name: str, cols: list[str]
) -> tuple[bool, str | None]:
    """Test if a test column name is in a list of column names."""
    if col_name not in cols:
        msg = f"Column {col_name} not found in DataFrame. "
        msg += "Please provide a valid column name:"
        for c in cols:
            msg += f"\n- {c}"
        return False, msg
    return True, None
