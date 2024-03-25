"""Produces a dictionary mapping dates to values."""

from __future__ import annotations
import polars as pl
import datetime


def _get_value_map(
    lf: pl.LazyFrame, date_col: str, x_col: str
) -> dict[datetime.date, float]:
    """Produce a dictionary mapping dates to values.

    This is used to map the list of dates to a list of values.

    Parameters
    ----------
    lf : pl.LazyFrame
        The input LazyFrame.
    date_col : str
        The name of the date column.
    x_col : str
        The name of the value column.

    Returns
    -------
    dict[datetime.date, float]
        A dictionary mapping dates to values.
    """
    return dict(lf.select([pl.col(date_col), pl.col(x_col)]).collect().rows())
