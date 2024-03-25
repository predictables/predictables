"""Find the continuous numeric columns in a DataFrame, and validate they are formatted as expected."""

from __future__ import annotations
from functools import wraps
import pandas as pd
import polars as pl
import typing

from predictables.util.src._get_column_dtype import get_column_dtype


def validate_cts_col(
    func: typing.Callable,
    test_df: pd.DataFrame | pl.DataFrame | pl.LazyFrame | None = None,
) -> typing.Callable:
    """Validate that a passed name represents a continuous numeric column in a DataFrame.

    First scans the pandas/polars DataFrame/LazyFrame to find numeric columns,
    and then validates that the column is formatted as expected. If not,
    it raises an error.

    If no data frame is passed, this decorator will check whether the DataFrame
    is the first argument of the function. If it is, it will use that DataFrame
    to validate the column. If not, it will raise an error.

    Parameters
    ----------
    func : typing.Callable
        The function to be wrapped.
    test_df : pd.DataFrame | pl.DataFrame | pl.LazyFrame | None
        The DataFrame to be validated. Defaults to None. If None, will check the
        first argument of the function to see if it is a DataFrame.

    Returns
    -------
    typing.Callable
        The wrapped function.

    Raises
    ------
    ValueError
        If the column is not validated as a continuous numeric column.
    """

    @wraps(func)
    def wrapper(*args, **kwargs) -> typing.Callable:
        # Get the DataFrame
        has_df = test_df is not None
        df = test_df if has_df else args[0]

        # If no DataFrame is found, raise an error
        if not isinstance(df, (pd.DataFrame, pl.DataFrame, pl.LazyFrame)):
            raise ValueError(
                "No dataframe passed to the validator, and no dataframe found in the"
                " function arguments. Either load a dataframe directly into the validator,"
                " or pass a dataframe as the first argument of the function."
            )

        # Depending on the DataFrame type, run the appropriate validation function
        df = validate_pd(df) if isinstance(df, pd.DataFrame) else validate_pl(df)

        return func(*args, **kwargs)

    return wrapper


def validate_pd(df: pd.DataFrame) -> pd.DataFrame:
    """Validate the date columns in a pandas DataFrame.

    Loops over the columns, and raises an error if a column that should be a date
    is not formatted as a date.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame.

    Returns
    -------
    pd.DataFrame
        The same DataFrame, just validated to have date columns formatted correctly.

    Raises
    ------
    ValueError
        If a column that should be a date is not formatted as a date.
    """
    for col in df.columns:
        if (get_column_dtype(df[col]) == "date") & (
            not pd.api.types.is_datetime64_any_dtype(df[col])
        ):
            raise ValueError(error_msg(df, col))

    return df


def validate_pl(df: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame | pl.LazyFrame:
    """Validate the date columns in a polars DataFrame.

    Loops over the columns, and raises an error if a column that should be a date
    is not formatted as a date.

    Parameters
    ----------
    df : pl.DataFrame | pl.LazyFrame
        The input DataFrame.

    Returns
    -------
    pl.DataFrame | pl.LazyFrame
        The same DataFrame, just validated to have date columns formatted correctly.

    Raises
    ------
    ValueError
        If a column that should be a date is not formatted as a date.
    """
    for col in df.columns:
        if (get_column_dtype(df[col]) == "date") & (
            df[col].dtype not in (pl.Date, pl.Datetime)
        ):
            raise ValueError(error_msg(df, col))

    return df


def error_msg(col: str) -> str:
    """Return an error message for a column that should be a date."""
    return f"Column {col} is not formatted as a date:\n\n{col}"
