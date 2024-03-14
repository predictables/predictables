from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl


def to_pl_df(
    df: pd.DataFrame | pl.DataFrame | pl.LazyFrame | pd.Series | pl.Series | np.ndarray,
) -> pl.DataFrame:
    """Convert to a polars dataframe."""
    if isinstance(df, pd.DataFrame):
        return pl.from_pandas(df)
    elif isinstance(df, pl.DataFrame):
        return df
    elif isinstance(df, pl.LazyFrame):
        return df.collect()
    elif isinstance(df, pd.Series):
        return pl.from_pandas(df.to_frame())
    elif isinstance(df, pl.Series):
        return pl.DataFrame({f"{df.name}": df})
    elif isinstance(df, np.ndarray):
        return pl.from_pandas(pd.DataFrame(df))
    else:
        raise TypeError(f"df must be a pandas or polars dataframe. Got {type(df)}.")


def to_pl_lf(
    df: pd.DataFrame | pl.DataFrame | pl.LazyFrame | pd.Series | pl.Series | np.ndarray,
) -> pl.LazyFrame:
    """Convert to a polars lazy frame."""
    if isinstance(df, pd.DataFrame):
        return pl.from_pandas(df).lazy()
    elif isinstance(df, pl.DataFrame):
        return df.lazy()
    elif isinstance(df, pl.LazyFrame):
        return df
    elif isinstance(df, pd.Series):
        return pl.from_pandas(df.to_frame()).lazy()
    elif isinstance(df, pl.Series):
        return pl.DataFrame({f"{df.name}": df}).lazy()
    elif isinstance(df, np.ndarray):
        return pl.from_pandas(pd.DataFrame(df)).lazy()
    else:
        raise TypeError("df must be a pandas or polars dataframe.")


def to_pl_s(s: pd.Series | pl.Series) -> pl.Series:
    """Convert to a polars series."""
    if isinstance(s, pd.Series):
        if s.dtype.name == "category":
            df = pl.from_pandas(s.to_frame()).lazy()
            col = df.columns[0]

            df = df.select([pl.col(col).cast(pl.Utf8).cast(pl.Categorical).name.keep()])
            return df.select(col).collect()[col]
        return pl.Series(name=s.name if s.name else None, values=s.values)
    elif isinstance(s, pl.Series):
        return s
    elif isinstance(s, np.ndarray):
        if s.ndim > 1:
            raise ValueError(f"s must be a 1-dimensional array - s.shape = {s.shape}")
        return pl.Series(s)
    elif isinstance(s, list):
        return pl.Series(s)
    else:
        raise TypeError(f"s must be a pandas or polars series, not {type(s)}.")
