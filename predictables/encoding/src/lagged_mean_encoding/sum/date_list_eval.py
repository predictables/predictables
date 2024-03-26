"""Calculate the rolling sum of a list column of dates."""

import polars as pl
from predictables.encoding.src.lagged_mean_encoding.sum.get_value_map import (
    _get_value_map,
)


# Functions returning expressions to calculate the min, max, and count
# of the date list column
def date_list_eval(date: pl.Expr, x: pl.Expr) -> pl.Expr:
    """Take a dictionary mapping dates to values and returns a function that can be used to map a list of dates to a list of values.

    This function is used in conjunction with `_handle_date_list` to map the list
    of dates to a list of values. The rolling sum is then calculated by summing up
    the values in the list column of values.

    Parameters
    ----------
    value_map : dict
        A dictionary mapping dates to values.

    Returns
    -------
    function
        A function that can be used to map a list of dates to a list of values.
    """
    return (
        # If the date is within the window:
        pl.when(
            (pl.col("date_list") >= pl.col("min_date"))
            & (pl.col("date_list") <= pl.col("max_date"))
        )
        # Then map the date to the value:
        .then(pl.col("date_list").dt.to_string("%m/%d/%Y").replace(old=date, new=x))
        # Otherwise, map the date to 0.0:
        .otherwise(pl.lit("0.0"))
        .str.to_decimal()
        .fill_null(0.0)
        .alias("value_list")
    )


def _handle_date_list(
    lf: pl.LazyFrame, x_col: str, date_col: str, index_col: str
) -> pl.LazyFrame:
    """Map the list of dates to a list of values and calculate the rolling sum.

    This function is used in conjunction with `_date_list_eval` to map the list of
    dates to a list of values. The rolling sum is then calculated by summing up the
    values in the list column of values.

    Parameters
    ----------
    lf : pl.LazyFrame
        The input LazyFrame.
    x_col : str
        The name of the value column (eg the column that will be summed).
    date_col : str
        The name of the date column.
    index_col : str
        The name of the index column.

    Returns
    -------
    pl.LazyFrame
        A LazyFrame with the rolling sum.
    """
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
        .group_by(index_col)
        .agg(pl.sum("value_list").alias(f"rolling_{x_col}"))
        # Resort the data to the original order
        .sort(index_col)
    )


def _date_list_eval(value_map: dict) -> pl.Expr:
    """Take a dictionary mapping dates to values and returns a function that can be used to map a list of dates to a list of values.

    This function is used to map the list column of dates to a list column of values. The rolling sum
    is then calculated by summing up the values in the list column of values.

    Parameters
    ----------
    value_map : dict
        A dictionary mapping dates to values.

    Returns
    -------
    pl.Expr
        An expression that can be used to map a list of dates to a list of values. This is a polars
        expression that can be used in the with_columns method of a LazyFrame.
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
