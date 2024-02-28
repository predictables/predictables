import datetime
import polars as pl
from typing import Dict

from predictables.util import validate_lf


@validate_lf
def dynamic_rolling_sum(
    lf: pl.LazyFrame,
    x_col: str,
    date_col: str,
    index_col: str = "index",
    offset: int = 30,
    window: int = 360,
) -> pl.LazyFrame:

    # Check to ensure there is an index column:
    if index_col not in lf.columns:
        raise ValueError(
            f"Index column {index_col} not found in LazyFrame. Please provide a valid index column."
        )
    # Data frame with the row index as a column, to resort the data
    # to the original order after the rolling sum
    lf_order = lf.select([pl.col(index_col).alias("index")])

    # Format columns in input LazyFrame
    lf_ = _format_date_col(lf, date_col)
    lf_ = _format_value_col(lf_, x_col)

    # Create a list column with the dates to be used for the
    # rolling sum, optionally grouping by categorical columns
    # if they are provided
    lf_ = _get_date_list_col(lf_, date_col, offset, window)

    # Map the list of dates to a list of values, censoring
    # any dates that are not in the original LazyFrame to 0
    lf_ = _handle_date_list(lf_, x_col, date_col)

    # The rolling sum is then the sum of that censored list
    lf_final = lf_.with_row_index().select(
        [pl.col("index"), pl.col(f"rolling_{x_col}").sum().alias(f"{x_col}_rolling")]
    )

    # Resort the data to the original order
    return lf_order.join(lf_final, on="index", how="left")


@validate_lf
def _format_date_col(lf: pl.LazyFrame, date_col: str) -> pl.LazyFrame:
    """
    Takes a LazyFrame and the name of the date column, and returns a LazyFrame
    with the date column correctly formatted as a Date.
    """
    return lf.with_columns([pl.col(date_col).cast(pl.Date).name.keep()])


@validate_lf
def _format_value_col(lf: pl.LazyFrame, value_col: str) -> pl.LazyFrame:
    """
    Takes a LazyFrame and the name of the value column, and returns a LazyFrame
    with the value column correctly formatted as a Float64.
    """
    return lf.with_columns([pl.col(value_col).cast(pl.Float64).name.keep()])


@validate_lf
def _get_date_list_col(
    lf: pl.LazyFrame,
    date_col: str,
    offset: int = 30,
    window: int = 360,
) -> pl.LazyFrame:
    """
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
    return lf.with_columns(
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


@validate_lf
def _get_value_map(
    lf: pl.LazyFrame,
    date_col: str,
    x_col: str,
) -> Dict[datetime.date, float]:
    """
    Produces a dictionary mapping dates to values. This is used to map the
    list of dates to a list of values.

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
    Dict[datetime.date, float]
        A dictionary mapping dates to values.
    """
    return {
        d: v for d, v in lf.select([pl.col(date_col), pl.col(x_col)]).collect().rows()
    }


@validate_lf
def _handle_date_list(lf: pl.LazyFrame, x_col: str, date_col: str) -> pl.LazyFrame:
    return (
        lf
        # Melt the elements of each list in the date_list column to create
        # a new row for each date in each list. Note also this will duplicate
        # the values in the other columns, including the row index and the
        # date column
        .explode("date_list")
        # Create a new column with the value for each date in the list
        .with_columns(_date_list_eval(_get_value_map(lf, date_col, x_col)))
        # Drop the date list column -- at this point we have substituted
        # each date in the list with the sum value for that date
        .drop("date_list")
        # Sum up the values by index (representing the original row order)
        # and date
        .group_by(["index", date_col]).agg(
            pl.sum("value_list").alias(f"rolling_{x_col}")
        )
        # Resort the data to the original order
        .sort("index")
    )


def _date_list_eval(value_map: dict) -> pl.Expr:
    """
    Takes a dictionary mapping dates to values and returns a function that
    can be used to map a list of dates to a list of values.
    """
    return (
        pl.when(pl.col("date_list").is_in(list(value_map.keys())))
        .then(
            pl.col("date_list").replace(
                old=pl.Series(value_map.keys()), new=pl.Series(value_map.values())
            )
        )
        .otherwise(pl.lit(0))
        .alias("value_list")
    )
