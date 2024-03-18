"""
This module performs correlation matrix analysis on the data.

Functions
---------
_correlation_matrix : Calculates the correlation matrix for the data.
_correlation_matrix_plot : Plots the correlation matrix.
_correlation_matrix_elimination : Eliminates the features with high correlation.
"""

from __future__ import annotations

from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl

from predictables.src._utils import _to_numpy, _to_pandas


def _pearson_correlation_matrix(
    data: Union[pd.DataFrame, pl.DataFrame, pl.LazyFrame], method: str = "pearson"
) -> pd.DataFrame:
    """
    Calculates the correlation matrix for the data.

    Parameters
    ----------
    data : Union[pd.DataFrame, pl.DataFrame,  pl.LazyFrame]
        The data to be analyzed.
    method : str, optional
        The method to calculate the correlation matrix. The default is 'pearson'. No other
        methods are currently supported.

    Returns
    -------
    pd.DataFrame
        The correlation matrix.
    """
    # Check if one of the disallowed types:
    if isinstance(data, pd.Series | pl.Series):
        raise TypeError(
            f"Input data type {type(data)} not supported. \n\
Please use one of the following types: \n\
    - pandas.DataFrame \n\
    - polars.DataFrame \n\
    - polars.LazyFrame"
        )

    # Check that if either a numpy array or list, that it is 2D
    if isinstance(data, np.ndarray):
        if len(data.shape) != 2:
            raise ValueError(f"Input data must be 2D, but is {len(data.shape)}D.")
    elif isinstance(data, list) and len(data) != 2:
        raise ValueError(f"Input data must be 2D, but is {len(data)}D.")

    # If data is a DataFrame, grab the column names
    if isinstance(data, pd.DataFrame):
        columns = data.columns.tolist()
    elif isinstance(data, pl.DataFrame | pl.LazyFrame):
        columns = data.columns
    else:
        columns = [f"x{i}" for i in range(len(data[0]))]

    # Convert to numpy array
    data = _to_numpy(data)

    # Calculate correlation matrix
    if method == "pearson":
        corr = np.corrcoef(data, rowvar=False)
    else:
        raise ValueError(
            f"Correlation method {method} not supported. \
Please use 'pearson'. I don't know why this is even an option."
        )

    # Convert to pandas DataFrame
    corr = _to_pandas(corr, "df")

    # Set column names
    corr.columns = columns
    corr.index = columns

    return corr


def _correlation_matrix(
    data: Union[pd.DataFrame, pl.DataFrame, pl.LazyFrame], method: str = "pearson"
) -> pd.DataFrame:
    """
    Calculates the correlation matrix for the data.

    Parameters
    ----------
    data : Union[pd.DataFrame, pl.DataFrame,  pl.LazyFrame]
        The data to be analyzed.
    method : str, optional
        The method to calculate the correlation matrix. The default is 'pearson'. No other
        methods are currently supported.

    Returns
    -------
    pd.DataFrame
        The correlation matrix.
    """
    return _pearson_correlation_matrix(data, method)


def _correlation_matrix_plot(
    data: Union[
        pd.DataFrame, pd.Series, pl.DataFrame, pl.Series, pl.LazyFrame, np.ndarray, list
    ],
    method: str = "pearson",
    ax: plt.Axes = None,
) -> plt.Axes:
    """
    Plots the correlation matrix. If an axis is passed, the plot will be added to the
    axis. Otherwise, a new figure will be created.

    Either way, the axis will be returned.

    Parameters
    ----------
    data : Union[pd.DataFrame, pd.Series, pl.DataFrame, pl.Series, pl.LazyFrame, np.ndarray, list]
        The data to be analyzed.
    method : str, optional
        The method to calculate the correlation matrix. The default is 'pearson',
        but accepts 'spearman' and 'kendall' as well.
    ax : plt.Axes, optional
        The axis to add the plot to. If None, a new figure will be created.

    Returns
    -------
    plt.Axes
        The axis containing the plot.
    """
    # Calculate correlation matrix
    corr = _correlation_matrix(data, method)

    # Plot correlation matrix
    if ax is None:
        fig, ax = plt.subplots()
    ax.matshow(corr, cmap="coolwarm")
    ax.set_xticks(np.arange(len(corr.columns)))
    ax.set_yticks(np.arange(len(corr.index)))
    ax.set_xticklabels(corr.columns)
    ax.set_yticklabels(corr.index)
    ax.tick_params(axis="x", rotation=90)
    ax.set_title(f"Correlation Matrix ({method.capitalize()})")

    return ax


def _highly_correlated_variables(
    data: Union[
        pd.DataFrame, pd.Series, pl.DataFrame, pl.Series, pl.LazyFrame, np.ndarray, list
    ],
    method: str = "pearson",
    threshold: float = 0.9,
) -> pl.LazyFrame:
    """
    Returns a list of tuples of highly correlated variables. Only one of the variables
    in each tuple will be kept.

    Parameters
    ----------
    data : Union[pd.DataFrame, pd.Series, pl.DataFrame, pl.Series, pl.LazyFrame, np.ndarray, list]
        The data to be analyzed.
    method : str, optional
        The method to calculate the correlation matrix. The default is 'pearson',
        but accepts 'spearman' and 'kendall' as well.
    threshold : float, optional
        The threshold for correlation. The default is 0.9.

    Returns
    -------
    pl.LazyFrame
        A list of tuples of highly correlated variables.
    """
    # Calculate correlation matrix
    corr = _correlation_matrix(data, method)

    # Get highly correlated variables:

    # 1. Absolute value
    corr = corr.abs()

    # 2. Lower triangle
    corr = corr.where(np.tril(np.ones(corr.shape), k=-1).astype(bool))

    # 3. Unstack (to get pairs)
    corr = corr.unstack()

    # 4. Remove NaNs
    corr = corr.dropna()

    # 5. Filter by threshold
    corr = corr[corr.gt(threshold)]

    # 6. Reset index
    corr = corr.reset_index()

    # 7. Rename columns
    corr = corr.rename({"level_0": "var1", "level_1": "var2", 0: "corr"}, axis=1)

    # 8. Sort by correlation
    corr = corr.sort_values(by="corr", ascending=False)

    # 9. Reset index
    corr = corr.reset_index(drop=True)

    # 10. Convert to list of tuples
    corr = list(corr["var1 var2".split()].to_records(index=False))

    return corr
