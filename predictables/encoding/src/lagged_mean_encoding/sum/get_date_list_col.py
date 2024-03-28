"""Generate a LazyFrame with a new column containing a list of dates."""

import polars as pl
import datetime
from predictables.util import to_pl_lf
from predictables.encoding.src.lagged_mean_encoding.sum.validation import (
    validate_offset,
    validate_window,
    validate_date_col,
)


@validate_offset
@validate_window
@validate_date_col(1, 0)
def _get_date_list_col(
    lf: pl.LazyFrame, date_col: str, offset: int = 30, window: int = 360
) -> pl.LazyFrame:
    """Generate a LazyFrame with a new column containing a list of dates.

    Takes a LazyFrame and the name of the date column, and optionally an
    integer representing the offset and window, and returns a LazyFrame with
    a new column containing a list of dates. This list of dates is used to
    calculate the rolling sum:

    1.  The list of dates is used to filter the original LazyFrame to only
        include the rows that are within the window.
    2.  The the incremental sum of the value column is calculated on that
        filtered LazyFrame.

    Parameters
    ----------
    lf : pl.LazyFrame
        The input LazyFrame.
    date_col : str
        The name of the date column.
    offset : int, default 30
        The number of days to offset the rolling sum by.
    window : int, default 360
        The number of days to include in the rolling sum.

    Returns
    -------
    pl.LazyFrame
        A LazyFrame with a new column containing a struct with keys
        for a list of dates.
    """
    # Return a list of dates for each row
    return to_pl_lf(lf).with_columns(
        [
            pl.date_ranges(
                start=pl.col(date_col)
                - datetime.timedelta(days=offset)
                - datetime.timedelta(days=window)
                + datetime.timedelta(days=1),
                end=pl.col(date_col) - datetime.timedelta(days=offset),
                interval="1d",
            ).alias("date_list")
        ]
    )
