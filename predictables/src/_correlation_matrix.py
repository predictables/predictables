"""
This module performs correlation matrix analysis on the data.

Functions
---------
_correlation_matrix : Calculates the correlation matrix for the data.
_correlation_matrix_plot : Plots the correlation matrix.
_correlation_matrix_elimination : Eliminates the features with high correlation.
"""

from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl

from predictables.src._utils import _to_numpy, _to_pandas


def _pearson_correlation_matrix(
    data: Union[pd.DataFrame, pl.DataFrame, pl.LazyFrame],
    method: str = "pearson",
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
    elif isinstance(data, np.ndarray):
        if len(data.shape) != 2:
            raise ValueError(
                f"Input data must be 2D, but is {len(data.shape)}D."
            )
    elif isinstance(data, list):
        if len(data) != 2:
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
    data: Union[pd.DataFrame, pl.DataFrame, pl.LazyFrame],
    method: str = "pearson",
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


# @vectorize(['float64(float64, float64)'], target='parallel')
# def _numpy_point_biserial_correlation(binary: np.ndarray,
#                                       continuous: np.ndarray) -> float:
#     """
#     Calculates the point biserial correlation coefficient between a binary variable and
#     a continuous variable using pure numpy. This is done to take advantage of numba.

#     For a continuous variable Y and a binary variable X, the point biserial correlation
#     coefficient is calculated as follows:

#     r_pb = ((mean(Y|X=1) - mean(Y|X=0)) / std(Y)) * sqrt(count(Y|X=1) * count(Y|X=0) / (count(Y) ** 2))

#     where:
#         - std(Y) is the unbiased standard deviation estimator:
#             std(Y) = sqrt(sum((Y - mean(Y)) ** 2) / (count(Y) - 1))

#         - count(Y) is the total number of observations of Y
#         - count(Y|X=i) is the number of observations of Y where X=i
#         - mean(Y|X=i) is the mean of Y where X=i

#     Parameters
#     ----------
#     binary : np.ndarray
#         The binary variable. Only a single column is allowed.
#     continuous : np.ndarray
#         The continuous variable. Only a single column is allowed.

#     Returns
#     -------
#     float
#         The point biserial correlation coefficient.
#     """
#     # Remove values of -999 from the binary variable, and the same indices from the
#     # continuous variable
#     Y = continuous[binary != -999]
#     X = binary[binary != -999]

#     # Calculate the mean, count, sd as needed
#     mean_Y_X1 = np.mean(Y[X == 1])
#     mean_Y_X0 = np.mean(Y[X == 0])
#     count_Y_X1 = np.sum(X == 1)
#     count_Y_X0 = np.sum(X == 0)
#     count_Y = len(Y)
#     sd_Y = np.sqrt(np.sum((Y - np.mean(Y)) ** 2) / (count_Y - 1))
#     bias_correction = np.sqrt(count_Y_X1 * count_Y_X0 / (count_Y ** 2))

#     # Calculate the point biserial correlation coefficient
#     r_pb = ((mean_Y_X1 - mean_Y_X0) / sd_Y) * bias_correction

#     return r_pb

# @jit(nopython=True, parallel=True)
# def _point_biserial_correlation_numba(binary: np.ndarray[np.float64],
#                                       binary_len: int,
#                                       continuous: np.ndarray[np.float64],
#                                       continuous_len: int) -> np.ndarray:
#     """
#     Loops through the binary variables (rows) and continuous variables (columns) and
#     calculates the point biserial correlation coefficient.

#     Uses the numba library to speed up the loop calculation.

#     Parameters
#     ----------
#     binary : np.ndarray
#         The binary data to be analyzed.
#     binary_len : int
#         The number of columns in the binary data.
#     continuous : np.ndarray
#         The continuous data to be analyzed.
#     continuous_len : int
#         The number of columns in the continuous data.

#     Returns
#     -------
#     np.ndarray
#         The correlation matrix.
#     """
#     # Create a numpy array from the binary data (to use numba)
#     corr_np = np.array([[np.float64(0) \
#                             for i in range(continuous_len)] \
#                                 for j in range(binary_len)])

#     # Loop through the binary variables
#     for i in range(binary_len):
#         # Loop through the continuous variables
#         for j in range(continuous_len):
#             # Calculate the point-biserial correlation coefficient using numpy
#             # in order to use numba
#             b = np.array([np.float64(b) for b in binary[:, i]])
#             c = np.array([np.float64(c) for c in continuous[:, j]])

#             # Remove values of -999 from the binary variable, and the same indices from the
#             # continuous variable
#             Y = c[b != -999]
#             X = b[b != -999]

#             # Calculate the mean, count, sd as needed
#             mean_Y_X1 = np.mean(Y[X == 1])
#             mean_Y_X0 = np.mean(Y[X == 0])
#             count_Y_X1 = np.sum(X == 1)
#             count_Y_X0 = np.sum(X == 0)
#             count_Y = len(Y)
#             sd_Y = np.sqrt(np.sum((Y - np.mean(Y)) ** 2) / (count_Y - 1))
#             bias_correction = np.sqrt(count_Y_X1 * count_Y_X0 / (count_Y ** 2))

#             # Calculate the point biserial correlation coefficient
#             r_pb = ((mean_Y_X1 - mean_Y_X0) / sd_Y) * bias_correction

#             corr_np[i, j] = r_pb

#     return corr_np

# def _point_biserial_correlation_numpy(binary: np.ndarray[np.float64],
#                                       binary_len: int,
#                                       continuous: np.ndarray[np.float64],
#                                       continuous_len: int) -> np.ndarray:
#     """
#     Loops through the binary variables (rows) and continuous variables (columns) and
#     calculates the point biserial correlation coefficient.

#     Uses the numba library to speed up the loop calculation.

#     Parameters
#     ----------
#     binary : np.ndarray
#         The binary data to be analyzed.
#     binary_len : int
#         The number of columns in the binary data.
#     continuous : np.ndarray
#         The continuous data to be analyzed.
#     continuous_len : int
#         The number of columns in the continuous data.

#     Returns
#     -------
#     np.ndarray
#         The correlation matrix.
#     """
#     # Create a numpy array from the binary data (to use numba)
#     corr_np = np.array([[np.float64(0) \
#                             for i in range(continuous_len)] \
#                                 for j in range(binary_len)])

#     # Loop through the binary variables
#     for i in range(binary_len):
#         # Loop through the continuous variables
#         for j in range(continuous_len):
#             # Calculate the point-biserial correlation coefficient using numpy
#             # in order to use numba
#             b = np.array([np.float64(b) for b in binary[:, i]])
#             c = np.array([np.float64(c) for c in continuous[:, j]])

#             # Remove values of -999 from the binary variable, and the same indices from the
#             # continuous variable
#             Y = c[b != -999]
#             X = b[b != -999]

#             # Calculate the mean, count, sd as needed
#             mean_Y_X1 = np.mean(Y[X == 1])
#             mean_Y_X0 = np.mean(Y[X == 0])
#             count_Y_X1 = np.sum(X == 1)
#             count_Y_X0 = np.sum(X == 0)
#             count_Y = len(Y)
#             sd_Y = np.sqrt(np.sum((Y - np.mean(Y)) ** 2) / (count_Y - 1))
#             bias_correction = np.sqrt(count_Y_X1 * count_Y_X0 / (count_Y ** 2))

#             # Calculate the point biserial correlation coefficient
#             r_pb = ((mean_Y_X1 - mean_Y_X0) / sd_Y) * bias_correction

#             corr_np[i, j] = r_pb

#     return corr_np

# def _point_biserial_correlation_polars(binary: pl.DataFrame,
#                                       continuous: pl.DataFrame) -> np.ndarray:
#     """
#     Loops through the binary variables (rows) and continuous variables (columns) and
#     calculates the point biserial correlation coefficient.

#     Uses the numba library to speed up the loop calculation.

#     Parameters
#     ----------
#     binary : np.ndarray
#         The binary data to be analyzed.
#     continuous : np.ndarray
#         The continuous data to be analyzed.

#     Returns
#     -------
#     np.ndarray
#         The correlation matrix.
#     """
#     # Create a pl.LazyFrame for the correlation matrix with a row for each
#     # binary / continuous combination
#     corr = pl.DataFrame({'binary': binary.columns.repeat(len(continuous.columns)),
#                          'continuous': np.tile(continuous.columns, len(binary.columns))})\
#              .lazy()

#     # Add a column for the inputs of the point biserial correlation coefficient
#     corr = corr.with_columns([
#         # Calculate the sum of the continuous variable where the binary variable is 1
#         pl.col('continuous_sum_X1') << pl.lazy(lambda df: df['binary'] * df['continuous']).sum(),
#     ])


# # Calculate the mean, count, sd as needed
# mean_Y_X1 = np.mean(Y[X == 1])
# mean_Y_X0 = np.mean(Y[X == 0])
# count_Y_X1 = np.sum(X == 1)
# count_Y_X0 = np.sum(X == 0)
# count_Y = len(Y)
# sd_Y = np.sqrt(np.sum((Y - np.mean(Y)) ** 2) / (count_Y - 1))
# bias_correction = np.sqrt(count_Y_X1 * count_Y_X0 / (count_Y ** 2))

# # Calculate the point biserial correlation coefficient
# r_pb = ((mean_Y_X1 - mean_Y_X0) / sd_Y) * bias_correction


# return corr_np

# def _point_biserial_correlation_matrix(data: Union[pd.DataFrame,
#                                                    pl.DataFrame,
#                                                    pl.LazyFrame]) -> pd.DataFrame:
#     """
#     Calculates the correlation matrix between a continuous variable and a binary variable.

#     For a continuous variable Y and a binary variable X, the point biserial correlation
#     coefficient is calculated as follows:

#     r_pb = (mean(Y|X=1) - mean(Y|X=0)) / std(Y)

#     This is a special case of the Pearson correlation coefficient.

#     Parameters
#     ----------
#     data : Union[pd.DataFrame, pl.DataFrame,  pl.LazyFrame]
#         The data to be analyzed. Must contain at least one binary variable and at least one
#         continuous variable.

#     Returns
#     -------
#     pd.DataFrame
#         The correlation matrix.
#     """
#     # 1. Convert to pandas DataFrame if necessary
#     if isinstance(data, pl.DataFrame | pl.LazyFrame):
#         data = _to_pandas(data, 'df')

#     # 2. Create separate tables with binary and continuous variables
#     # Binary variables will be categorical, with at most three categories:
#     #   1. "0"
#     #   2. "1"
#     #   3. "-999" / "missing"
#     #
#     # Start with all categorical variables
#     binary = data[_select_binary_columns(data)]

#     # Get the continuous variables
#     continuous = data.select_dtypes(include='number')

#     print(f'binary: {binary.shape}, continuous: {continuous.shape}')
#     print(f"binary: {binary}")
#     print(f"continuous: {continuous}")

#     # 3. Calculate the correlation matrix
#     corr = _point_biserial_correlation_numba(binary.to_numpy(np.float64),
#                                              binary.shape[1],
#                                              continuous.to_numpy(np.float64),
#                                              continuous.shape[1])

#     return corr


def _correlation_matrix_plot(
    data: Union[
        pd.DataFrame,
        pd.Series,
        pl.DataFrame,
        pl.Series,
        pl.LazyFrame,
        np.ndarray,
        list,
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
        pd.DataFrame,
        pd.Series,
        pl.DataFrame,
        pl.Series,
        pl.LazyFrame,
        np.ndarray,
        list,
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
    corr = corr.rename(
        {"level_0": "var1", "level_1": "var2", 0: "corr"}, axis=1
    )

    # 8. Sort by correlation
    corr = corr.sort_values(by="corr", ascending=False)

    # 9. Reset index
    corr = corr.reset_index(drop=True)

    # 10. Convert to list of tuples
    corr = list(corr["var1 var2".split()].to_records(index=False))

    return corr
