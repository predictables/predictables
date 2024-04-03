"""Return the minimum date in the date list column."""

import polars as pl


def min_date() -> pl.Expr:
    """Return the minimum date in the date list column."""
    return (
        pl.col("date_list").list.eval(pl.element().min()).list.first().alias("min_date")
    )
