"""
This module provides functionality to impute missing values in dataframes with the mode.
It can work with both `pandas` and `polars` data structures, converting them into `polars`
DataFrames for processing. It is designed to impute values specifically for categorical columns.

Functions:
    - `_is_numeric_dtype`: Check if a polars data type is numeric.
    - `_check_if_categorical`: Check if a column in the dataframe is categorical.
    - `_impute_col_with_mode`: Impute missing values in a column with the mode.
    - `impute_with_mode`: Apply mode imputation to all categorical columns in the dataframe.

The main entry point is the `impute_with_mode` function, which takes a dataframe and applies
mode imputation to all eligible columns.

Example:
    >>> import pandas as pd
    >>> df = pd.DataFrame({'A': ['a', 'b', None, 'b'], 'B': [1, 2, None, 4]})
    >>> imputed_df = impute_with_mode(df)
    >>> imputed_df
"""

import pandas as pd
import polars as pl
from typing import Union
from PredicTables.util import to_pl_df


def _is_numeric_dtype(dtype) -> bool:
    """
    Check if a polars data type is numeric.

    :param dtype: Polars data type of a dataframe column.
    :type dtype: polars.datatypes.DataType
    :return: `True` if the data type is numeric, `False` otherwise.
    :rtype: bool
    """
    numeric_dtypes = [
        pl.Int64,
        pl.Float64,
        pl.Int32,
        pl.Float32,
        pl.Int16,
        pl.Int8,
        pl.UInt8,
        pl.UInt16,
        pl.UInt32,
        pl.UInt64,
    ]
    return dtype in numeric_dtypes


def _check_if_categorical(df: pl.DataFrame, col: str) -> bool:
    """
    Check if a column in the dataframe is categorical.

    :param df: A polars DataFrame.
    :type df: pl.DataFrame
    :param col: The name of the column to check.
    :type col: str
    :raises ValueError: If the specified column does not exist in the dataframe.
    :return: `True` if the column is categorical, `False` if it is numeric.
    :rtype: bool
    """
    if col not in df.columns:
        raise ValueError(f"Column {col} does not exist in the dataframe.")
    return not _is_numeric_dtype(df[col].dtype)


def _impute_col_with_mode(df: pl.DataFrame, col: str) -> pl.DataFrame:
    """
    Impute missing values in a column with the mode.

    :param df: A polars DataFrame with potential missing values.
    :type df: pl.DataFrame
    :param col: The name of the column for which to impute missing values.
    :type col: str
    :return: A new DataFrame with missing values in the specified column imputed with the mode.
    :rtype: pl.DataFrame
    """
    if _check_if_categorical(df, col):
        mode = (
            df.group_by(col)
            .agg(pl.count().alias("count"))
            .sort("count")
            .reverse()[col]
            .head(1)[0]
        )
        # Use fill_null on the Series object, not on Expr

        # if there are multiple modes, return the first
        if _check_multiple_modes(df, col):
            mode = mode

        df = df.with_columns(
            pl.when(pl.col(col).is_null())
            .then(pl.lit(mode))
            .otherwise(pl.col(col))
            .alias(col)
        )

    return df


def _check_multiple_modes(df: pl.DataFrame, col: str) -> bool:
    """Check if the series has more than one mode.

    Args:
        series (pd.Series): The pandas Series to check for multiple modes.

    Returns:
        bool: True if there is more than one mode, False otherwise.
    """
    series = df.to_pandas()[col]
    modes = series.mode()
    return len(modes) > 1


def impute_with_mode(
    df: Union[pl.DataFrame, pd.DataFrame, pl.LazyFrame, pd.Series, pl.Series]
) -> pl.LazyFrame:
    """
    Impute missing values with the mode for categorical columns in the dataframe.

    :param df: A dataframe which can be a pandas DataFrame, Series, polars DataFrame, Series, or LazyFrame.
    :type df: Union[pl.DataFrame, pd.DataFrame, pl.LazyFrame, pd.Series, pl.Series]
    :return: A new DataFrame with missing values in categorical columns imputed with the mode.
    :rtype: pl.DataFrame
    """
    df = to_pl_df(df)
    n_rows = df.shape[0]

    for col in df.columns:
        # skip any columns that are all None
        n_nulls = df.to_pandas()[col].isna().sum()
        if n_rows == n_nulls:
            continue

        # if the column is not numeric, impute it
        elif df[col].dtype not in [
            pl.Int64,
            pl.Float64,
            pl.Int32,
            pl.Float32,
            pl.Int16,
            pl.Int8,
        ]:
            if df[col].dtype == pl.Categorical:
                df = df.with_columns([pl.col(col).cast(pl.Utf8).keep_name()])
            df = _impute_col_with_mode(df, col)

    return df.lazy()
