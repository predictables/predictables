"""
This module implements the Box-Cox transformation.

The Box-Cox transformation is a power transformation of the data that is
intended to stabilize the variance of the data, and in particular to lessen
the impact of outliers on the data. It depends on a parameter lambda, which
is chosen to minimize the variance of the data. The Box-Cox transformation
is defined as:

(y^{lambda} - 1) / lambda, if lambda != 0
log(y), if lambda == 0
"""

import pandas as pd
import polars as pl

from typing import Tuple, Union

from sklearn.preprocessing import PowerTransformer

def _calc_box_cox_transform(df: pl.LazyFrame, column:str, lambda_: float) -> pl.Series:
    """
    Calculate the Box-Cox transformation of a Series. This is a helper function
    for the Box-Cox transformation, and is not intended to be called directly.

    Parameters
    ----------
    df : pl.LazyFrame
        The DataFrame containing the Series to transform.
    column : str
        The name of the Series to transform.
    lambda_ : float
        The parameter lambda of the Box-Cox transformation.

    Returns
    -------
    pl.LazyFrame
        The DataFrame with the transformed Series added.
    """
    # Calculate the Box-Cox transformation
    if lambda_ == 0:
        df = df.with_columns([pl.col(column).log().alias(f"{column}_box_cox")])
    else:
        df = df.with_columns([((pl.col(column).pow(lambda_)) - 1) / lambda_\
                              .alias(f"{column}_box_cox")])
                              
    return df

def _box_cox_transform(data: Union[pd.DataFrame,
                                   pl.DataFrame,
                                   pl.LazyFrame],
                       column: str) -> Tuple[pl.LazyFrame, float]:
    """
    Calculate the Box-Cox transformation of the given data. The Box-Cox
    transformation is a power transformation of the data that is intended
    to stabilize the variance of the data, and in particular to lessen the
    impact of outliers on the data. It depends on a parameter lambda, which
    is estimated to minimize the variance of the data. The Box-Cox
    transformation is defined as:

    (y^{lambda} - 1) / lambda, if lambda != 0
    log(y), if lambda == 0

    Parameters
    ----------
    data : Union[pd.DataFrame,
                 pl.DataFrame,
                 pl.LazyFrame]
        The data to transform.
    column : str
        The name of the column to transform.

    Returns
    -------
    pl.LazyFrame
        The DataFrame with the transformed column added.
    float
        The estimated lambda parameter.
    """
    # Get the column from the DataFrame
    if isinstance(data, pd.DataFrame):
        df = pl.from_pandas(data)
    elif isinstance(data, pl.DataFrame):
        df = data
    elif isinstance(data, pl.LazyFrame):
        df = data
    else:
        raise TypeError("data must be a DataFrame or LazyFrame")

    # Calculate the Box-Cox transformation
    box_cox = PowerTransformer(method='box-cox', standardize=False)
    box_cox.fit(df[[column]])