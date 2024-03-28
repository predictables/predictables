"""Format the columns of a LazyFrame as either dates or floats."""

import polars as pl


def _format_date_col(lf: pl.LazyFrame, date_col: str) -> pl.LazyFrame:
    """Generate a LazyFrame with the date column correctly formatted as a Date.

    Takes a LazyFrame and the name of the date column, and returns a LazyFrame
    with the date column correctly formatted as a Date.
    """
    return lf.with_columns([pl.col(date_col).cast(pl.Date).name.keep()])


def _format_value_col(lf: pl.LazyFrame, value_col: str) -> pl.LazyFrame:
    """Generate a LazyFrame with the value column correctly formatted as a Float64.

    Takes a LazyFrame and the name of the value column, and returns a LazyFrame
    with the value column correctly formatted as a Float64.
    """
    return lf.with_columns([pl.col(value_col).cast(pl.Float64).name.keep()])
