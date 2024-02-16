# trunk-ignore-all(sourcery)
"""
This module contains the get_cv_folds function for generating cross-validation folds.
"""

from typing import Union

import pandas as pd
import polars as pl
from sklearn.model_selection import KFold  # type: ignore

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
        If True, returns list of indices for each fold; else returns a series
        with fold numbers.

    Returns
    -------
    Union[pl.Series, list]
        A series with fold numbers or a list of indices for each fold.
    """
    df = to_pd_df(df)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    return (
        list((train_idx, test_idx) for train_idx, test_idx in kf.split(df))
        if return_indices
        else pl.Series(kf.split(df)[1])
    )
