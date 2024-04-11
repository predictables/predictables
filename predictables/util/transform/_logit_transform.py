from __future__ import annotations

import polars as pl


def logit_transform(
    column_name: str, renamed_column_name: str | None = None
) -> pl.Expr:
    """Transform a variable using the logit of the variable.

    Note that because this function is intended to be used in a pipeline, it
    cannot directly validate the column name. Instead, it will return an
    expression that will be validated when the pipeline is executed.

    Parameters
    ----------
    column_name : str
        The name of the column to be logit-transformed.
    renamed_column_name : str, optional
        The name of the new column. If None, the column will be named
        f'logit[{column_name}]'.

    Returns
    -------
    pl.Expr
        An expression representing the logit-transformed column.
    """
    # Handle default value for renamed_column_name
    new_col_name = (
        f"logit[{column_name}]" if renamed_column_name is None else renamed_column_name
    )

    numerator = pl.col(column_name)
    denominator = pl.lit(1).sub(pl.col(column_name))
    odds = numerator.truediv(denominator)

    # Return the logit(column) expression (eg the log-odds ratio)
    return odds.log().alias(new_col_name)
