from typing import Union

import numpy as np
import pandas as pd
import polars as pl


def to_pd_df(df: Union[pd.DataFrame, pl.DataFrame, pl.LazyFrame]) -> pd.DataFrame:
    """
    Convert a dataframe to a pandas dataframe.
    """
    if isinstance(df, pd.DataFrame):
        return df
    elif isinstance(df, pl.DataFrame):
        return df.to_pandas()
    elif isinstance(df, pl.LazyFrame):
        return df.collect().to_pandas()
    elif isinstance(df, pd.Series):
        return df.to_frame()
    elif isinstance(df, pl.Series):
        return df.to_pandas().to_frame()
    elif isinstance(df, np.ndarray):
        return pd.DataFrame(df)
    else:
        raise TypeError(f"df must be a pandas or polars dataframe. Got {type(df)}.")


def to_pd_s(s: Union[pd.Series, pl.Series]) -> pd.Series:
    """
    Convert to a pandas series.
    """
    if isinstance(s, pd.Series):
        return s
    elif isinstance(s, pl.Series):
        if s.dtype == "category":
            return s.cast(pl.Utf8).to_pandas()[s.name].astype("category")
        else:
            return s.to_pandas()
    elif isinstance(s, np.ndarray):
        if s.ndim > 1:
            raise ValueError("s must be a 1D array.")
        return pd.Series(s.flatten())
    elif isinstance(s, list):
        return pd.Series(s)
    else:
        raise TypeError(f"s must be a pandas or polars series. Got {type(s)}.")


# git commit -m "added list to to_pd_s and to_pl_s acceptable types, and tests pass"
