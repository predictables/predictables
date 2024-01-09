import pandas as pd
import polars as pl

from typing import Union
from PredicTables.util import to_pl_lf


def group_and_aggregate(
    df: Union[pd.DataFrame, pl.DataFrame, pl.LazyFrame], lowest_grain: str, target: str
) -> pl.LazyFrame:
    """
    Groups the provided DataFrame by the column identified as the lowest grain,
    and aggregates the target column by taking its maximum value.

    Parameters
    ----------
    df : Union[pd.DataFrame, pl.DataFrame, pl.LazyFrame]
        The DataFrame to group and aggregate.
    lowest_grain : str
        The name of the column identifying the lowest grain.
    target : str
        The name of the target column to aggregate.

    Returns
    -------
    pl.LazyFrame
        The grouped and aggregated DataFrame.

    Raises
    ------
    ValueError
        If the target column is not numeric.
    """
    lf = to_pl_lf(df)
    return lf.select(
        [pl.col(lowest_grain), pl.col(target).max().over(lowest_grain).alias(target)]
    ).unique()
