import polars as pl


def remove_missing_rows(df, *columns):
    """
    Remove rows from a Polars DataFrame where any of the specified columns contain missing values.

    Parameters
    ----------
    df : pl.LazyFrame
        The input lazy Polars DataFrame.
    *columns : str
        Column names to check for missing values.

    Returns
    -------
    pl.LazyFrame
        A lazy Polars DataFrame with rows removed where specified columns contain missing values.
    """
    for col in columns:
        df = df.filter(pl.col(col).is_not_null())

    return df
