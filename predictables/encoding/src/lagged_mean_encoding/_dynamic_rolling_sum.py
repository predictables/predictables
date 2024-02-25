import polars as pl
from typing import List, Optional, Dict, Union
from datetime import datetime


def dynamic_rolling_sum(
    lf: pl.LazyFrame,
    x: str,
    date_col: str,
    category_cols: Optional[Union[str, List[str]]] = None,
    x_name: Optional[str] = None,
    every: str = "1d",
    period: str = "1y",
    offset: str = "-1mo",
) -> pl.Series:
    """
    Calculate the rolling sum for each row in the lazyframe.

    Parameters
    ----------
    lf : pl.LazyFrame
        The lazyframe to calculate the rolling sum for.
    x : str
        The name of the column to calculate the rolling sum for.
    date_col : str
        The name of the date column.
    category_cols : Union[str, List[str]], optional
        The name of the category column, or a list of names of category columns.
    x_name : str, optional
        The name of the new column to add to the lazyframe.
    every : str, optional
        The frequency of the rolling sum. Default is "1d".
    period : str, optional
        The period of the rolling sum. Default is "1y".
    offset : str, optional
        The offset of the rolling sum. Default is "-1mo".

    Returns
    -------
    pl.Series
        The rolling sum for each row in the lazyframe.
    """
    # Format the date and, optionally, category columns to ensure they
    # are in the correct format
    date_expr = _formatted_date_col(date_col)
    category_cols_ = (
        _formatted_category_cols(category_cols) if category_cols is not None else None
    )
    lf = lf.with_columns(
        [date_expr] + (category_cols_ if category_cols_ is not None else [])
    )

    # Get the lazyframe with the original row order preserved
    lf_order = _get_original_order(lf, date_col, category_cols)

    # Get the name of the new column to add to the lazyframe
    x_name = _get_x_name(x, x_name)

    # Calculate the rolling sum
    if category_cols is not None:
        sum_lf = _rolling_sum_categories(
            lf, x, date_col, category_cols, x_name, every, period, offset
        )
    else:
        sum_lf = _rolling_sum_no_categories(
            lf, x, date_col, x_name, every, period, offset
        )

    # Resort the lazyframe to the original order
    resorted_lf = sum_lf.join(lf_order, on="index").sort("index")

    return resorted_lf.select(pl.col(x_name)).to_series()


def _get_x_name(x: str, x_name: Optional[str]) -> str:
    """
    Get the name of the new column to add to the lazyframe.

    Parameters
    ----------
    x : str
        The name of the column to calculate the rolling sum for.
    x_name : str, optional
        The name of the new column to add to the lazyframe.

    Returns
    -------
    str
        The name of the new column to add to the lazyframe.
    """
    return f"{x}_rolling_sum" if x_name is None else x_name


def _get_date_map(lf: pl.LazyFrame, date_col: str) -> Dict[datetime, datetime]:
    """
    Returns a dictionary mapping dates to their reversed dates.

    Parameters
    ----------
    lf : pl.LazyFrame
        The lazyframe.
    date_col : str
        The name of the date column that will be used to create the date map.

    Returns
    -------
    Dict[datetime, datetime]
        A dictionary that maps dates to their reversed dates.
    """
    min_date = lf.select(pl.col(date_col).min()).collect().to_series()[0]
    max_date = datetime.today()
    date_range = pl.datetime_range(start=min_date, end=max_date).cast(pl.Date)
    date_df = pl.select(date_range.alias(date_col)).with_columns(
        [
            pl.col(date_col).cast(pl.Date).name.keep(),
            pl.col(date_col).cast(pl.Date).reverse().alias(f"{date_col}_reversed"),
        ]
    )
    date_map = {
        d20: d10 for d10, d20 in zip(date_df[date_col], date_df[f"{date_col}_reversed"])
    }

    return date_map


def _get_original_order(
    lf: pl.LazyFrame, date_col: str, category_cols: Optional[List[str]] = None
) -> pl.LazyFrame:
    """
    Returns a lazyframe with the original row order preserved.

    Parameters
    ----------
    lf : pl.LazyFrame
        The lazyframe.
    date_col : str
        The name of the date column.
    category_cols : List[str], optional
        The list of names of the categorical columns.

    Returns
    -------
    pl.LazyFrame
        The lazyframe with the original row order preserved.
    """
    date_map = _get_date_map(lf, date_col)
    return (
        lf.with_row_index()
        .with_columns(
            [
                # Add the reversed date column
                pl.col(date_col)
                .replace(date_map)
                .cast(pl.Date)
                .alias(f"{date_col}_reversed")
            ]
        )
        .sort(
            [f"{date_col}_reversed"]
            + (category_cols if category_cols is not None else [])
        )
        .select(pl.col("index"))
    )


def _formatted_category_cols(category_cols: Union[str, List[str]]) -> List[pl.Expr]:
    """
    Format the category columns to ensure they are in the correct format.

    Parameters
    ----------
    category_cols : Union[str, List[str]]
        The name of the category column, or a list of names of category columns.

    Returns
    -------
    List[pl.Expr]
        A list of expressions for the formatted category columns.

    Notes
    -----
    Even if only one category column is passed, it will be returned as a list
    of expressions.
    """
    cat_cols = [category_cols] if isinstance(category_cols, str) else category_cols
    return [
        pl.col(col).cast(pl.Utf8).cast(pl.Categorical).name.keep() for col in cat_cols
    ]


def _formatted_date_col(date_col: Union[str, pl.Expr]) -> pl.Expr:
    """
    Format the date column to ensure it is in the correct format.

    Parameters
    ----------
    date_col : Union[str, pl.Expr]
        The name of the date column, or an expression for a date column.

    Returns
    -------
    pl.Expr
        An expression for the formatted date column.

    """
    expr = pl.col(date_col) if isinstance(date_col, str) else date_col
    return expr.cast(pl.Date)


def _reversed_date_col(lf: pl.LazyFrame, date_col: str) -> pl.Expr:
    """
    Add a column for the reversed date.

    Parameters
    ----------
    lf : pl.LazyFrame
        The lazyframe containing the date column.
    date_col : str
        The name of the date column.

    Returns
    -------
    pl.Expr
        The reversed date column.
    """
    date_map = _get_date_map(lf, date_col)
    reversed_expr = pl.col(date_col).replace(date_map)
    return _formatted_date_col(reversed_expr)


def _group_by_no_categories(
    lf: pl.LazyFrame,
    reversed_date_expr: pl.Expr,
    x: str,
    x_name: Optional[str],
    every: str = "1d",
    period: str = "1y",
    offset: str = "-1mo",
) -> pl.LazyFrame.group_by.LazyGroupBy:
    """
    Group the lazyframe by the reversed date column. This function is used when there
    are no category columns to group by.

    Parameters
    ----------
    lf : pl.LazyFrame
        The lazyframe to group by.
    reversed_date_expr : pl.Expr
        The expression for the reversed date column.
    x : str
        The name of the column to calculate the rolling sum for.
    x_name : str, optional
        The name of the new column to add to the lazyframe.
    every : str, optional
        The frequency of the rolling sum. Default is "1d".
    period : str, optional
        The period of the rolling sum. Default is "1y".
    offset : str, optional
        The offset of the rolling sum. Default is "-1mo".

    Returns
    -------
    pl.LazyFrame.group_by.LazyGroupBy
        The grouped lazyframe.
    """
    return lf.sort(reversed_date_expr).group_by_dynamic(
        reversed_date_expr,
        every=every,
        period=period,
        offset=offset,
        start_by="datapoint",
        check_sorted=False,
    )


def _group_by_categories(
    lf: pl.LazyFrame,
    reversed_date_expr: pl.Expr,
    x: str,
    x_name: Optional[str],
    category_cols: List[str],
    every: str = "1d",
    period: str = "1y",
    offset: str = "-1mo",
) -> pl.LazyFrame.group_by.LazyGroupBy:
    """
    Group the lazyframe by the reversed date column and the category columns.
    This function is used when there are category columns to group by.

    Parameters
    ----------
    lf : pl.LazyFrame
        The lazyframe to group by.
    reversed_date_expr : pl.Expr
        The expression for the reversed date column.
    x : str
        The name of the column to calculate the rolling sum for.
    x_name : str, optional
        The name of the new column to add to the lazyframe.
    category_cols : List[str]
        The list of names of the categorical columns.
    every : str, optional
        The frequency of the rolling sum. Default is "1d".
    period : str, optional
        The period of the rolling sum. Default is "1y".
    offset : str, optional
        The offset of the rolling sum. Default is "-1mo".

    Returns
    -------
    pl.LazyFrame.group_by.LazyGroupBy
        The grouped lazyframe.
    """
    return (
        lf.with_columns(_formatted_category_cols(category_cols))
        .sort(reversed_date_expr, *category_cols)
        .group_by_dynamic(
            reversed_date_expr,
            every=every,
            period=period,
            offset=offset,
            by=category_cols,
            start_by="datapoint",
            check_sorted=False,
        )
    )


def _rolling_sum_no_categories(
    lf: pl.LazyFrame,
    x: str,
    date_col: str,
    x_name: Optional[str],
    every: str = "1d",
    period: str = "1y",
    offset: str = "-1mo",
) -> pl.LazyFrame:
    """
    Calculate the rolling sum for each row in the lazyframe. This function is used when there are no category columns to group by.

    Parameters
    ----------
    lf : pl.LazyFrame
        The lazyframe to calculate the rolling sum for.
    x : str
        The name of the column to calculate the rolling sum for.
    date_col : str
        The name of the date column.
    x_name : str, optional
        The name of the new column to add to the lazyframe.
    every : str, optional
        The frequency of the rolling sum. Default is "1d".
    period : str, optional
        The period of the rolling sum. Default is "1y".
    offset : str, optional
        The offset of the rolling sum. Default is "-1mo".

    Returns
    -------
    pl.LazyFrame
        The lazyframe with the rolling sum added as a new column.
    """
    # Get the lazyframe with the original row order preserved
    lf_order = _get_original_order(lf, date_col)

    # Add a column for the reversed date
    reversed_date_expr = _reversed_date_col(lf, date_col)

    # Use the reverse-sorted date to calculate the rolling sum
    lfgby = _group_by_no_categories(
        lf_order, reversed_date_expr, x, x_name, every, period, offset
    )

    # Calculate the rolling sum
    return lfgby.agg(pl.sum(x).name.keep())


def _rolling_sum_categories(
    lf: pl.LazyFrame,
    x: str,
    date_col: str,
    category_cols: List[str],
    x_name: Optional[str],
    every: str = "1d",
    period: str = "1y",
    offset: str = "-1mo",
) -> pl.LazyFrame:
    """
    Calculate the rolling sum for each row in the lazyframe. This function is used when there are category columns to group by.

    Parameters
    ----------
    lf : pl.LazyFrame
        The lazyframe to calculate the rolling sum for.
    x : str
        The name of the column to calculate the rolling sum for.
    date_col : str
        The name of the date column.
    category_cols : List[str]
        The list of names of the categorical columns.
    x_name : str, optional
        The name of the new column to add to the lazyframe.
    every : str, optional
        The frequency of the rolling sum. Default is "1d".
    period : str, optional
        The period of the rolling sum. Default is "1y".
    offset : str, optional
        The offset of the rolling sum. Default is "-1mo".

    Returns
    -------
    pl.LazyFrame
        The lazyframe with the rolling sum added as a new column.
    """
    # Get the lazyframe with the original row order preserved
    lf_order = _get_original_order(lf, date_col, category_cols)

    # Add a column for the reversed date
    reversed_date_expr = _reversed_date_col(lf, date_col)

    # Use the reverse-sorted date to calculate the rolling sum
    lfgby = _group_by_categories(
        lf_order, reversed_date_expr, x, x_name, category_cols, every, period, offset
    )

    # Calculate the rolling sum
    return lfgby.agg(pl.sum(x).name.keep())
