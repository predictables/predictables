import pandas as pd
import polars as pl

from PredicTables.src._utils import _to_polars

def _iqr(df: pl.LazyFrame, col: str) -> float:
    """
    Helper function to calculate the interquartile range (IQR) of a column in a LazyFrame.
    """
    # Convert to LazyFrame if necessary
    df = _to_polars(df, 'lf')

    # Calculate IQR
    iqr = df.select(pl.col(col).quantile(0.75) - pl.col(col).quantile(0.25)).collect().item()

    return iqr

def _tukeys_fences(df: pl.LazyFrame, col: str, k: float = 1.5) -> tuple:
    """
    Function to calculate Tukey's fences for a column in a LazyFrame.
    """
    # Convert to LazyFrame if necessary
    df = _to_polars(df, 'lf')

    # Calculate IQR
    iqr = _iqr(df, col)

    # Calculate Tukey's fences
    lower_fence = df.select(pl.col(col).quantile(0.25) - k * iqr).collect().item()
    upper_fence = df.select(pl.col(col).quantile(0.75) + k * iqr).collect().item()

    return lower_fence, upper_fence

def tukeys_outliers(df: pl.LazyFrame,
                  col: str,
                  k: float = 1.5) -> pl.LazyFrame:
    """
    Function to calculate Tukey's fences a column in a LazyFrame.

    Parameters
    ----------
    df : pl.LazyFrame
        Polars LazyFrame.
    col : str
        Column name.    
    k : float, optional
        Tukey's fences multiplier, by default 1.5.

    Returns
    -------
    pl.LazyFrame
        Polars LazyFrame filtered to only include outliers, as defined by Tukey's fences.
        
    """
    # Convert to LazyFrame if necessary
    df = _to_polars(df, 'lf')

    # Calculate Tukey's fences
    lower_fence, upper_fence = _tukeys_fences(df, col, k)

    # Filter to only include outliers
    df = df.filter((pl.col(col) < lower_fence).or_(
        (pl.col(col) > upper_fence)
    ))

    # Return filtered LazyFrame
    return df