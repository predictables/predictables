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
    if isinstance(s, pd.Series) | isinstance(s, pl.Series) | isinstance(s, np.ndarray):
        # If the series is empty, raise a value error
        if s.shape[0] == 0:
            return pd.Series(s)

        # If the series is a pandas series, convert to a polars series
        if isinstance(s, pd.Series):
            return s
        elif isinstance(s, pl.Series):
            # if the data type is categorical, first convert to string, then to pandas, then back to categorical
            if s.dtype == "category":
                return s.astype(str).to_pandas().astype("category")
            else:
                arr = s.to_numpy()  ## This prevents the
                return pd.Series(arr)
        elif isinstance(s, np.ndarray):
            return pd.Series(s)
        else:
            raise TypeError(f"s must be a pandas or polars series. Got {type(s)}.")
    else:
        raise TypeError(f"s must be a pandas or polars series. Got {type(s)}.")
