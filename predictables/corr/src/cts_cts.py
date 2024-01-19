from typing import Union

import pandas as pd
import polars as pl

from predictables.util import get_column_dtype, to_pd_df, to_pd_s


def calc_continuous_continuous_corr(
    *args: Union[pd.Series, pl.Series, pd.DataFrame, pl.DataFrame, pl.LazyFrame],
) -> Union[float, pd.DataFrame]:
    """
    Calculates the correlation either between two continuous variables or between
    all pairs of continuous variables in a data frame.

    Parameters
    ----------
    *args : Union[pd.Series, pl.Series, pd.DataFrame, pl.DataFrame, pl.LazyFrame]
        The arguments to pass to the appropriate function.

        - If one argument is passed, it is assumed to be a data
          frame or some other similar data structure.
        - If two arguments are passed, they are assumed to be two
          continuous variables.

    Returns
    -------
    Union[float, pd.DataFrame]
        The correlation between two continuous variables or between all pairs of
        continuous variables in a data frame.
    """
    if len(args) == 1:
        return calc_continuous_continuous_corr_df(args[0])
    elif len(args) == 2:
        return calc_continuous_continuous_corr_series(args[0], args[1])
    else:
        raise TypeError(
            f"Invalid number of arguments: Must be 1 or 2, but got {len(args)}"
        )


def calc_continuous_continuous_corr_df(
    df: Union[pd.DataFrame, pl.DataFrame, pl.LazyFrame],
) -> pd.DataFrame:
    """
    Calculates the correlation between all pairs of continuous variables in the

    Parameters
    ----------
    df : Union[pd.DataFrame, pl.DataFrame, pl.LazyFrame]
        The dataset to analyze.

    Returns
    -------
    pd.DataFrame
        A dataframe containing the correlation between all pairs of continuous
        variables in the dataset.
    """
    dataset = to_pd_df(df)

    # make sure all columns are numeric - drop non-numeric columns
    for col in dataset.columns:
        if get_column_dtype(dataset[col]) != "continuous":
            dataset = dataset.drop(columns=[col])
            print(f"Warning: dropping non-continuous column {col}")

    return dataset.corr()


def calc_continuous_continuous_corr_series(
    s1: Union[pd.Series, pl.Series], s2: Union[pd.Series, pl.Series]
) -> float:
    """
    Calculates the correlation between two continuous variables.

    Parameters
    ----------
    s1 : Union[pd.Series, pl.Series]
        The first continuous variable.
    s2 : Union[pd.Series, pl.Series]
        The second continuous variable.

    Returns
    -------
    float
        The correlation between the two continuous variables.
    """
    # convert to pandas series
    s1 = to_pd_s(s1)
    s2 = to_pd_s(s2)

    # make sure both series are numeric
    if get_column_dtype(s1) != "continuous":
        raise TypeError(f"s1 is `{get_column_dtype(s1)}`, not continuous")
    if get_column_dtype(s2) != "continuous":
        raise TypeError(f"s2 is `{get_column_dtype(s2)}`, not continuous")

    return s1.corr(s2)
