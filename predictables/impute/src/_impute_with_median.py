from typing import Union

import pandas as pd
import polars as pl

from predictables.util import to_pl_lf, get_column_dtype


def _check_if_numeric(
    df: Union[pl.DataFrame, pd.DataFrame, pl.LazyFrame, pd.Series, pl.Series],
    col: str,
) -> bool:
    """
    Check if a column is numeric. This has been refactored to simply use
    the predictables.util.get_column_dtype function.
    """
    if isinstance(df, pd.DataFrame):
        c = df[col]
    elif isinstance(df, pl.DataFrame):
        c = df[col]
    elif isinstance(df, pl.LazyFrame):
        c = df.collect()[col]
    elif isinstance(df, pd.Series):
        c = df
    elif isinstance(df, pl.Series):
        c = df
    else:
        raise ValueError("The dataframe type is not supported")
    return get_column_dtype(c) in ["continuous", "integer", "float"]


def _impute_col_with_median(
    df: Union[pl.DataFrame, pd.DataFrame, pl.LazyFrame, pd.Series, pl.Series],
    col: str,
) -> pl.Series:
    """
    Impute missing values with the median of the column.

    Parameters
    ----------
    df : Union[pl.DataFrame, pd.DataFrame, pl.LazyFrame, pd.Series, pl.Series]
        A dataframe. Will be coerced to a polars lazy frame.
    col : str
        A column name.

    Returns
    -------
    pl.Series
        A column of a dataframe with missing values imputed with the median.

    Examples
    --------
    >>> df = pl.DataFrame({'A': [1, 2, None, 4]})
    >>> df.select(pl.col('A').median().alias('median')).collect()['median'][0]
    2
    """
    # Convert to a polars lazy frame
    if isinstance(df, (pd.DataFrame, pl.DataFrame, pl.LazyFrame)):
        df = to_pl_lf(df)
    elif isinstance(df, (pd.Series, pl.Series)):
        df = df.to_frame()
    else:
        raise ValueError("The dataframe type is not supported")

    # If the column is numeric, impute with the median
    return df[str(pl.col(col))].fill_null(
        df.select(pl.col(col).median().alias("median")).collect()["median"][0]
    )
    # else:
    #     return df


def impute_with_median(
    df: Union[pl.DataFrame, pd.DataFrame, pl.LazyFrame, pd.Series, pl.Series],
) -> pl.LazyFrame:
    """
    Loop through all the columns in a dataframe and impute missing values with the
    median.

    Parameters
    ----------
    df : Union[pl.DataFrame, pd.DataFrame, pl.LazyFrame, pd.Series, pl.Series]
        A dataframe. Will be coerced to a polars lazy frame.

    Returns
    -------
    pl.LazyFrame
        A dataframe with missing values imputed with the median from each column.

    Examples
    --------
    >>> df = pl.DataFrame({'A': [1, 2, None, 4]})

    >>> df.select(pl.col('A').median().alias('median')).collect()['median'][0]
    2

    >>> impute_with_median(df).collect()
    shape: (4, 1)
    ╭─────╮
    │ A   │
    │ --- │
    │ i64 │
    ╞═════╡
    │ 1   │
    ├─────┤
    │ 2   │
    ├─────┤
    │ 2   │
    ├─────┤
    │ 4   │
    ╰─────╯
    """
    # Convert to a polars lazy frame
    df = to_pl_lf(df)

    # If the column is empty, just return the dataframe
    if df.collect().shape[0] == 0:
        return df

    # Loop through each column and impute with the median
    for col in df.columns:
        # If the column is numeric, impute with the median
        if _check_if_numeric(df, col):
            df = _impute_col_with_median(df, col)
        else:
            continue

    return df
