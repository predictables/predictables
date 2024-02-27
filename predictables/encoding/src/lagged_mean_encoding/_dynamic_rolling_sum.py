import datetime
import polars as pl
from predictables.encoding.src.lagged_mean_encoding._validation import validate
from typing import List, Optional, Dict


@validate
def dynamic_rolling_sum(
    lf: pl.LazyFrame,
    x_col: str,
    date_col: str,
    categorical_cols: Optional[List[str]] = None,
    offset: int = 30,
    window: int = 360,
) -> pl.Series:

    # Format columns in input LazyFrame
    lf_ = _format_date_col(lf, date_col)
    lf_ = _format_value_col(lf_, x_col)

    # Create a list column with the dates to be used for the
    # rolling sum, optionally grouping by categorical columns
    # if they are provided
    lf_ = _get_date_list_col(lf_, date_col, categorical_cols, offset, window)

    # Get the mapping from dates to values - this is
    # how we turn the list of dates into a rolling sum
    value_map = _get_value_map(lf_, date_col, x_col)

    # Map the list of dates to a list of values, censoring
    # any dates that are not in the original LazyFrame to 0
    lf_ = lf_.with_columns([_date_list_eval(value_map).alias("rolling_x")])

    # Optionally, group by the categorical columns if they are provided
    if categorical_cols is not None:
        print("Grouping by categorical columns")

    # The rolling sum is then the sum of that censored list
    return lf_.select(pl.col("rolling_x").sum().alias(x_col)).collect().to_series()


@validate
def _format_date_col(lf: pl.LazyFrame, date_col: str) -> pl.LazyFrame:
    """
    Takes a LazyFrame and the name of the date column, and returns a LazyFrame
    with the date column correctly formatted as a Date.
    """
    return lf.with_columns([pl.col(date_col).cast(pl.Date).name.keep()])


@validate
def _format_value_col(lf: pl.LazyFrame, value_col: str) -> pl.LazyFrame:
    """
    Takes a LazyFrame and the name of the value column, and returns a LazyFrame
    with the value column correctly formatted as a Float64.
    """
    return lf.with_columns([pl.col(value_col).cast(pl.Float64).name.keep()])


@validate
def _get_date_list_col(
    lf: pl.LazyFrame,
    date_col: str,
    categorical_cols: Optional[List[str]] = None,
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
    2.  Optionally, the filtered LazyFrame is grouped by the categorical
        column names provided.
    3.  The the incremental sum of the value column is calculated on that
        filtered (and possibly grouped) LazyFrame.

    Parameters
    ----------
    lf : pl.LazyFrame
        The input LazyFrame.
    date_col : str
        The name of the date column.
    categorical_cols : Optional[List[str]], default None
        The names of the categorical columns.
    offset : int, default 30
        The number of days to offset the rolling sum by.
    window : int, default 360
        The number of days to include in the rolling sum.

    Returns
    -------
    pl.LazyFrame
        A LazyFrame with a new column containing a struct with keys
        for a list of dates, and the categorical columns if they are
        provided.

    Notes
    -----
    Each string passed to the categorical_cols parameter must be the name
    of a column in the input LazyFrame. The keys of the struct will be
    derived from the names of the columns in the input LazyFrame.

    The list of dates is used to calculate the rolling sum. The rolling sum
    is the sum of the value column for the dates in the list, offset by the
    number of days specified in the offset parameter, including the
    number of days specified in the window parameter, optionally grouped
    by the categorical columns if they are provided.
    """
    # Get the list of dates (for each row) to be used for the rolling sum
    date_list = [
        pl.date_ranges(
            start=pl.col(date_col)
            - datetime.timedelta(days=offset)
            - datetime.timedelta(days=window)
            + datetime.timedelta(days=1),
            end=pl.col(date_col) - datetime.timedelta(days=offset),
            interval="1d",
        ).alias("date_list")
    ]

    # Turn into a dictionary that will become the output struct
    expr_dict = {f"{date_col}": date_list}

    # Optionally group by the categorical columns
    if categorical_cols is not None:
        for col in categorical_cols:
            # Filter the LazyFrame to only include the rows whose dates
            # correspond to the dates in the list of dates for each row
            def filter_dt(dt):
                return lf.filter([pl.col(date_col) == dt])

            # Add a new key-value pair to the dictionary for each
            # categorical column, by finding the unique values for
            # that column for each date in the list of dates
            expr_dict[col] = [
                # Select the unique values for the categorical column
                filter_dt(dt).select(pl.col(col).unique()).collect().to_list()
                # For each date in the date_list for that row
                for dt in date_list
            ]

    # Create a new column with the struct
    return lf.with_columns([pl.struct(expr_dict).alias("groupby")])


@validate
def _get_value_map(
    lf: pl.LazyFrame,
    date_col: str,
    x_col: str,
    categorical_cols: Optional[List[str]] = None,
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


def _date_list_eval(value_map: dict) -> pl.Expr:
    """
    Takes a dictionary mapping dates to values and returns a function that
    can be used to map a list of dates to a list of values.
    """
    return (
        pl.col("groupby")
        .struct.field("date_list")
        .list.eval(  # Take the list of dates
            # For each date in the list, check if it's in the dictionary
            pl.when(pl.element().is_in(list(value_map.keys())))
            # If it is, map to the value for that date
            .then(value_map[pl.element()])
            # If it's not, map to 0
            .otherwise(pl.lit(0))
        )
    )


# def _cat_list_eval(col: str) -> pl.Expr:
#     """
#     Takes the list of lists corresponding to a categorical column and returns an expression
#     of the same length, but with a 1 if the row's category is in the list, and a 0 if it's not.
#     """

#     def eval_list(list_):
#         return [1 if cat in list_ else 0 for cat in list_]

#     struct = pl.col("groupby").struct.field(col)
#     return struct.list.eval()
