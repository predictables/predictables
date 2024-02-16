from typing import Union

import pandas as pd
import polars as pl

from predictables.impute.src._impute_with_median import impute_with_median
from predictables.impute.src._impute_with_mode import impute_with_mode
from predictables.util import to_pl_lf


def initial_impute(
    df: Union[pl.DataFrame, pd.DataFrame, pl.LazyFrame, pd.Series, pl.Series],
) -> pl.LazyFrame:
    """
    Loop through all the columns in a dataframe and impute missing values with
    the median or mode depending on the column type.

    Parameters
    ----------
    df : Union[pl.DataFrame, pd.DataFrame, pl.LazyFrame, pd.Series, pl.Series]
        A dataframe. Will be coerced to a polars lazy frame.

    Returns
    -------
    pl.LazyFrame
        A dataframe with missing values imputed with the median or mode from
        each column.
    """
    df = (
        to_pl_lf(df)
        if isinstance(df, (pd.DataFrame, pl.DataFrame, pl.LazyFrame))
        else to_pl_lf(df.to_frame())
    )

    # Loop through each column and impute with the median or mode
    return impute_with_mode(impute_with_median(df))
