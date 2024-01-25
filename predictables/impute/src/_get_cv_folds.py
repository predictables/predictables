"""
This module contains the get_cv_folds function for generating cross-validation folds.
"""

from typing import Union

import pandas as pd
import polars as pl
from sklearn.model_selection import KFold

from predictables.util import to_pd_df


def get_cv_folds(
    df: Union[pd.DataFrame, pl.DataFrame, pl.LazyFrame],
    n_folds: int = 5,
    return_indices: bool = False,
) -> Union[pl.Series, list]:
    """
    Get the cross-validation folds for each row in the dataframe.

    Parameters
    ----------
    df : Union[pd.DataFrame, pl.DataFrame, pl.LazyFrame]
        The dataframe to get the folds for.
    n_folds : int
        The number of folds to use. Default is 5-fold cross-validation.
    return_indices : bool
        If True, returns list of indices for each fold; else returns a series with fold numbers.

    Returns
    -------
    Union[pl.Series, list]
        A series with fold numbers or a list of indices for each fold.


    :return: A series with fold numbers or a list of indices for each fold.
    """
    if not isinstance(n_folds, int):
        raise TypeError("n_folds must be an integer.")
    if n_folds < 1:
        raise ValueError("n_folds must be greater than 0.")

    # Assuming to_pd_df efficiently converts dataframe types
    df = to_pd_df(df)
    kf = KFold(n_splits=n_folds)

    if return_indices:
        return [(train_idx, test_idx) for train_idx, test_idx in kf.split(df)]
    else:
        fold_series = pd.Series([0] * len(df))
        for fold_number, (_, test_idx) in enumerate(kf.split(df)):
            fold_series.iloc[test_idx] = fold_number
        return fold_series
