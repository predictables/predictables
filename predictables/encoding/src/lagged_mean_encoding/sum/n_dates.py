"""Return the number of dates in the date list column."""

import polars as pl


def n_dates() -> pl.Expr:
    """Return the number of dates in the date list column.

    Returns
    -------
    pl.Expr
        The number of dates in the date list column.
    """
    return pl.col("date_list").list.len().alias("n_dates")
