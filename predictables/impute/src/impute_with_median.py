from typing import Union

import pandas as pd
import polars as pl
import polars.selectors as cs

from predictables.util import to_pl_lf


def _check_if_numeric(
    df: Union[pl.DataFrame, pd.DataFrame, pl.LazyFrame, pd.Series, pl.Series], col: str
) -> bool:
    """Check if a column is numeric.

    :param[Union[pl.DataFrame, pd.DataFrame, pl.LazyFrame, pd.Series, pl.Series]] df: A dataframe. Will be coerced to a polars lazy frame.
    :param[str] col: A column name.

    :return[bool]: True if the column is numeric, False otherwise.

    :example:
    >>> df = pl.DataFrame({'A': [1, 2, None, 4]})
    >>> _check_if_numeric(df, 'A')
    True
    """
    # Convert to a polars lazy frame
    df = to_pl_lf(df)

    # Check if the column is numeric
    return col in df.select(cs.numeric()).columns


def _impute_col_with_median(
    df: Union[pl.DataFrame, pd.DataFrame, pl.LazyFrame, pd.Series, pl.Series], col: str
) -> pl.Series:
    """Impute missing values with the median of the column.

    :param[Union[pl.DataFrame, pd.DataFrame, pl.LazyFrame, pd.Series, pl.Series]] df: A dataframe. Will be coerced to a polars lazy frame.
    :param[str] col: A column name.

    :return[pl.Series]: A column of a dataframe with missing values imputed with the median.

    :example:
    >>> df = pl.DataFrame({'A': [1, 2, None, 4]})
    >>> df.select(pl.col('A').median().alias('median')).collect()['median'][0]
    2
    """
    # Convert to a polars lazy frame
    df = to_pl_lf(df)

    # If the column is numeric, impute with the median
    # if _check_if_numeric(df, col):
    # median = df.select(pl.col(col).median().alias("median")).collect()["median"][0]
    return df.with_columns(
        pl.col(col).fill_null(
            df.select(pl.col(col).median().alias("median")).collect()["median"][0]
        )
    )
    # else:
    #     return df


def impute_with_median(
    df: Union[pl.DataFrame, pd.DataFrame, pl.LazyFrame, pd.Series, pl.Series],
) -> pl.LazyFrame:
    """Loop through all the columns in a dataframe and impute missing values with the median.

    :param[Union[pl.DataFrame, pd.DataFrame, pl.LazyFrame, pd.Series, pl.Series]] df: A dataframe. Will be coerced to a polars lazy frame.

    :return[pl.LazyFrame]: A dataframe with missing values imputed with the median from each column.
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
