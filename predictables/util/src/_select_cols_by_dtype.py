from typing import Union

import pandas as pd
import polars as pl

from predictables.util.src._get_column_dtype import get_column_dtype
from predictables.util.src._to_pd import to_pd_df


def select_cols_by_dtype(
    df: Union[pd.DataFrame, pl.DataFrame, pl.LazyFrame], dtype: str
) -> pd.DataFrame:
    """
    Returns a data frame containing only the columns of the specified dtype. Uses the
    set of data types defined in
    PredicTables.util.src.get_column_dtype.get_column_dtype():

    - "continuous"
        - "integer" (sub-type of "continuous")
    - "categorical"
        - "binary" (sub-type of "categorical")
    - "datetime"

    Parameters
    ----------
    df : Union[pd.DataFrame, pl.DataFrame, pl.LazyFrame]
        The data frame to select columns from.
    dtype : str
        The dtype to select columns by. Must be one of the following:
        - "continuous"
        - "categorical"
        - "datetime"
        - "integer"
        - "binary"

    Returns
    -------
    pd.DataFrame
        A data frame containing only the columns of the specified dtype.

    """
    # Ensure dtype is valid
    dtype = dtype.lower()
    if dtype not in [
        "continuous",
        "categorical",
        "datetime",
        "integer",
        "binary",
    ]:
        raise ValueError(
            "dtype must be one of the following: 'continuous', "
            f"'categorical', 'datetime', 'integer', 'binary', but got {dtype}"
        )

    # Get the dtype of each column
    col_dtypes = [
        get_column_dtype(to_pd_df(df).iloc[:, i]) for i in range(len(df.columns))
    ]

    # Get the names of the columns of the specified dtype
    cols = (
        to_pd_df(df)
        .iloc[:, [i for i, x in enumerate(col_dtypes) if x == dtype]]
        .columns.tolist()
    )

    # Return a df with only the columns of the specified dtype
    return to_pd_df(df)[cols]
