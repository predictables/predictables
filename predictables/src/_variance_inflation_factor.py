from __future__ import annotations

from typing import Union

import numpy as np
import pandas as pd
import polars as pl
from statsmodels.stats.outliers_influence import variance_inflation_factor  # type: ignore

from predictables.util import to_pd_df, tqdm


def _vif_i(
    data: Union[pd.DataFrame, pl.DataFrame, pl.LazyFrame, np.ndarray],
    col_i: Union[int, str],
) -> float:
    """
    Return the Variance Inflation Factor (VIF) Score for a given feature.

    Parameters
    ----------
    data : Union[pd.DataFrame, pl.DataFrame, pl.LazyFrame, np.ndarray]
        The data to calculate the VIF score for. Will be converted to a numpy array.
    col_i : Union[int, str]
        Either an integer or string representing the column/column index of the
        feature to calculate the VIF score for.

    Returns
    -------
    float
        The VIF score for the given feature.
    """
    # Convert to numpy array
    if isinstance(data, pd.DataFrame):
        col_idx = (
            data.columns.tolist().index(col_i) if isinstance(col_i, str) else col_i
        )
        exog = to_pd_df(data).to_numpy()
    elif isinstance(data, pl.DataFrame):
        col_idx = data.columns.index(col_i) if isinstance(col_i, str) else col_i
        exog = to_pd_df(data).to_numpy()
    elif isinstance(data, pl.LazyFrame):
        col_idx = (
            data.collect().columns.index(col_i) if isinstance(col_i, str) else col_i
        )
        exog = to_pd_df(data).to_numpy()
    elif isinstance(data, np.ndarray):
        col_idx = col_i if isinstance(col_i, int) else np.where(data[0] == col_i)[0][0]
        exog = data
    else:
        raise TypeError(
            f"Data type {type(data)} not supported.\n\
Please use one of the following types:\n\
    - pandas.DataFrame\n\
    - polars.DataFrame\n\
    - polars.LazyFrame\n\
    - numpy.ndarray"
        )

    # Calculate VIF score
    return variance_inflation_factor(exog, col_idx)


def _vif(
    data: Union[pd.DataFrame, pl.DataFrame, pl.LazyFrame, np.ndarray],
    show_progress: bool = True,
) -> pd.DataFrame:
    """
    Return the Variance Inflation Factor (VIF) Scores for all features in the data.

    The variance inflation factor (VIF) statistic is used to quantify the severity
    of multicollinearity in an ordinary least squares regression analysis. It provides
    an index that measures how much the variance of an estimated regression coefficient
    is increased because of collinearity. It is calculated as:

        VIF_i = 1 / (1 - R^2_i)

    where R^2_i is the coefficient of determination in a regression of the ith predictor
    on the remaining predictors.

    To calculate the VIF for the ith predictor:
        1. Regress the ith predictor on all the other predictors - can the remaining predictors
           predict the ith predictor?
        2. Calculate the R^2 of the regression from step 1. Here the R^2 represents the proportion
           of the variance of the ith predictor that can be predicted from the remaining predictors.
        3. Calculate the VIF_i = 1 / (1 - R^2_i)
        4. VIF values > 10 are considered high and are a sign of multicollinearity. VIF > 10
           implies that the variance of a coefficient is *at least 10 times larger* than it would
           be if that predictor was uncorrelated with the other predictors.

    Parameters
    ----------
    data : Union[pd.DataFrame, pl.DataFrame, pl.LazyFrame, np.ndarray]
        The data to calculate the VIF scores for. Will be converted to a numpy array.
    show_progress : bool, optional
        Whether to show a progress bar, by default True

    Returns
    -------
    pd.DataFrame
        A dataframe containing the VIF scores for all features in the data.
    """
    # Get a list of all the features
    if isinstance(data, pd.DataFrame):
        features = data.columns.tolist()
    elif isinstance(data, pl.DataFrame):
        features = data.columns
    elif isinstance(data, pl.LazyFrame):
        features = data.collect().columns
    elif isinstance(data, np.ndarray):
        features = list(map(str, range(data.shape[1])))
    else:
        raise TypeError(
            f"Data type {type(data)} not supported.\n\
Please use one of the following types:\n\
    - pandas.DataFrame\n\
    - polars.DataFrame\n\
    - polars.LazyFrame\n\
    - numpy.ndarray"
        )

    # Loop over the features, calculating the VIF score for each
    column = []
    vif_score = []
    for i in tqdm(features, disable=not show_progress):
        column.append(i)
        vif_score.append(_vif_i(data, i))

    # Return the VIF scores as a dataframe
    df = (
        pd.DataFrame({"feature": column, "vif_score": vif_score})
        .sort_values(by="vif_score", ascending=False)
        .reset_index(drop=True)
    )
    return df
