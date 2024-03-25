"""Find the date columns in a DataFrame, and validate they are formatted as expected."""

from __future__ import annotations
from functools import wraps
import pandas as pd
import polars as pl
import typing
from predictables.util.src._get_column_dtype import get_column_dtype


def validate_date_col(
    func: typing.Callable,
    test_df: pd.DataFrame | pl.DataFrame | pl.LazyFrame | None = None,
) -> typing.Callable:
    """Validate that a passed name represents a date column in a DataFrame.

    First scans the pandas/polars DataFrame/LazyFrame to find date columns,
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
        If the column is not found in the DataFrame.
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
        if (get_column_dtype(df[col]) == "datetime") & (
            not pd.api.types.is_datetime64_any_dtype(df[col])
        ):
            raise ValueError(error_msg(col))

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
        if (get_column_dtype(df[col]) == "datetime") & (
            df[col].dtype not in (pl.Date, pl.Datetime)
        ):
            raise ValueError(error_msg(col))

    return df


def error_msg(col: str) -> str:
    """Return an error message for a column that should be a date."""
    return f"Column {col} is not formatted as a date:\n\n{col}"


# def sas_date_regex(text: str) -> datetime.date: # noqa: ERA001
#     """Find all dates in the specified format. # noqa: ERA001

#     Looking for the SAS date format: # noqa: ERA001
#         - DDMMMYYYY # noqa: ERA001

#     For example: # noqa: ERA001
#         - 1/1/2022 => 01JAN2022 # noqa: ERA001
#         - 2/2/2023 => 02FEB2023 # noqa: ERA001

#     If a SAS date format is found, the text is returned  # noqa: ERA001

#     Parameters # noqa: ERA001
# noqa: ERA001
#     """ # noqa: ERA001
#     date_pattern = re.compile(r"\b\d{2}(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\d{4}\b", re.IGNORECASE) # noqa: ERA001

#     # Month name lookup dict # noqa: ERA001
#     month_names = { # noqa: ERA001
#         "jan": "01", # noqa: ERA001
#         "feb": "02", # noqa: ERA001
#         "mar": "03", # noqa: ERA001
#         "apr": "04", # noqa: ERA001
#         "may": "05", # noqa: ERA001
#         "jun": "06", # noqa: ERA001
#         "jul": "07", # noqa: ERA001
#         "aug": "08", # noqa: ERA001
#         "sep": "09", # noqa: ERA001
#         "oct": "10", # noqa: ERA001
#         "nov": "11", # noqa: ERA001
#         "dec": "12" # noqa: ERA001
#     } # noqa: ERA001
# noqa: ERA001
#     # Find the pattern in the text # noqa: ERA001
#     return date # noqa: ERA001
