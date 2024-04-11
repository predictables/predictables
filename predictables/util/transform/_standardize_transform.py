from __future__ import annotations

import polars as pl


def standardize_transform(
    column_name: str,
    renamed_column_name: str | None = None,
    group_by_column: str | None = None,
) -> pl.Expr:
    """Standardize a variable to have mean 0 and standard deviation 1.

    Note that because this function is intended to be used in a pipeline, it
    cannot directly validate the column name. Instead, it will return an
    expression that will be validated when the pipeline is executed.

    Parameters
    ----------
    column_name : str
        The name of the column to be standardized
    renamed_column_name : str, optional
        The name of the new column. If None, the column will be named
        f'standardized[{column_name}]'.
    group_by_column : str, optional
        The name of the column to group by before standardizing. If None, the
        column will be standardized without grouping.

    Returns
    -------
    pl.Expr
        An expression representing the standardized column.
    """
    # Handle default value for renamed_column_name
    new_col_name = (
        f"standardized[{column_name}]"
        if renamed_column_name is None
        else renamed_column_name
    )

    # Calculate the mean and standard deviation of the column
    # respecting the group_by_column if it is not None
    if group_by_column is not None:
        mean_ = pl.col(column_name).over(group_by_column).mean()
        std_ = pl.col(column_name).over(group_by_column).std()
    else:
        mean_ = pl.col(column_name).mean()
        std_ = pl.col(column_name).std()

    # Return the standardized column expression
    return (
        pl.when(std_ == 0)
        .then(pl.lit(0))
        .otherwise((pl.col(column_name) - mean_).truediv(std_).alias(new_col_name))
    )
