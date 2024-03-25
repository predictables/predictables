import polars as pl


def max_date() -> pl.Expr:
    """Return the maximum date in the date list column."""
    return (
        pl.col("date_list").list.eval(pl.element().max()).list.first().alias("max_date")
    )
