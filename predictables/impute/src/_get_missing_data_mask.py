"""
This module contains the get_missing_data_mask function for getting a mask of the missing data in a dataframe.

Andy Weaver
"""

from typing import Union

import pandas as pd
import polars as pl

from predictables.util import to_pl_lf


def get_missing_data_mask(
    df: Union[pd.DataFrame, pl.DataFrame, pl.LazyFrame],
) -> pl.LazyFrame:
    """
    Get a mask of the missing data in the dataframe.

    Parameters
    ----------
    df : Union[pd.DataFrame, pl.DataFrame, pl.LazyFrame]
        The dataframe to get the missing data mask for.

    Returns
    -------
    pl.LazyFrame
        A lazy frame of the same size as the input dataframe that contains True for missing data and False for non-missing data.

    Raises
    ------
    TypeError
        If df is not a pandas or polars dataframe.

    Examples
    --------
    >>> import polars as pl
    >>> from PredicTables.impute import get_missing_data_mask
    >>> from PredicTables.util import to_pl_lf
    >>> df = pl.DataFrame({'a': [1, 2, None, 4, 5]})
    >>> get_missing_data_mask(df).collect()

    shape: (5, 1)
    ╭─────╮
    │ a   │
    │ --- │
    │ i64 │
    ╞═════╡
    │ 0   │
    ├╌╌╌╌╌┤
    │ 0   │
    ├╌╌╌╌╌┤
    │ 1   │
    ├╌╌╌╌╌┤
    │ 0   │
    ├╌╌╌╌╌┤
    │ 0   │
    ╰─────╯
    """
    if not isinstance(df, (pd.DataFrame, pl.DataFrame, pl.LazyFrame)):
        raise TypeError("df must be a pandas or polars dataframe.")
    return to_pl_lf(df).with_columns(
        [pl.col(c).is_null().name.keep() for c in df.columns]
    )
