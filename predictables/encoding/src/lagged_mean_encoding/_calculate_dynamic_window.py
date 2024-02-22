import polars as pl
from typing import List, Optional


def calculate_dynamic_window(
    date_column: str,
    numerator_column: str,
    denominator_column: Optional[str] = None,
    categorical_columns: Optional[List[str]] = None,
) -> pl.Expr:
    """
    For each row in the lf, calculate the sum of the numerator and denominator
    for the year prior to the date in that row, segmented by the categorical columns.

    Assumes date_column is of type Date.

    Dynamically calculates the sum of the numerator and denominator for each row,
    considering only the data from the year prior to the date in that row.

    Parameters
    ----------
    date_column : str
        The name of the date column.
    categorical_columns : List[str]
        The list of names of the categorical columns.
    numerator_column : str
        The name of the numerator column.
    denominator_column : str
        The name of the denominator column.

    Returns
    -------
    pl.Expr
        Polars expression to add the calculated column to the lazyframe.

    :return: Polars expression to add the calculated column to the lazyframe.
    """
    # Create a filter expression to select rows within the window
    date_filter = _get_date_filter(date_column)

    # Create a groupby expression to segment the data by the categorical columns
    groupby_expr = (
        [pl.lit(1)]
        if categorical_columns is None
        else [pl.col(col) for col in categorical_columns]
    )

    # Create a sum expression to calculate the sum of the numerator and denominator
    num = (
        pl.when(date_filter)
        .then(pl.col(numerator_column))
        .otherwise(0)
        .sum()
        .over(groupby_expr)
    )
    denom = (
        pl.when(date_filter)
        .then(
            [pl.col(denominator_column)]
            if denominator_column is not None
            else [pl.lit(1)]
        )
        .otherwise(pl.lit(0))
        .sum()
        .over(groupby_expr)
    )

    # Create a division expression to calculate the mean
    ratio = pl.when(denom.eq(0)).then(pl.lit(0)).otherwise(num.truediv(denom))
    return ratio.alias(f"{numerator_column}:{denominator_column}_ratio")


def _get_date_filter(date_column: str) -> pl.Expr:
    """
    Create a filter expression to select rows within the window.

    Parameters
    ----------
    date_column : str
        The name of the date column.

    Returns
    -------
    pl.Expr
        The filter expression.
    """
    one_month_prior = pl.col(date_column).dt.offset_by("-1mo")
    thirteen_months_prior = pl.col(date_column).dt.offset_by("-13mo")
    return (pl.col(date_column) >= thirteen_months_prior) & (
        pl.col(date_column) <= one_month_prior
    )
