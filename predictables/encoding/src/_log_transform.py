from __future__ import annotations

import polars as pl


def log_transform(column_name: str, renamed_column_name: str | None = None) -> pl.Expr:
    """Transform a variable using the log of 1 + the variable.

    Note that because this function is intended to be used in a pipeline, it
    cannot directly validate the column name. Instead, it will return an
    expression that will be validated when the pipeline is executed.

    Parameters
    ----------
    column_name : str
        The name of the column to be log-transformed.
    renamed_column_name : str, optional
        The name of the new column. If None, the column will be named
        f'log[1+{column_name}]'.

    Returns
    -------
    pl.Expr
        An expression representing the log-transformed column.
    """
    # Handle default value for renamed_column_name
    new_col_name = (
        f"log[{column_name}]" if renamed_column_name is None else renamed_column_name
    )

    # Return the log(1 + column) expression
    return pl.lit(1).add(pl.col(column_name)).log().alias(new_col_name)
