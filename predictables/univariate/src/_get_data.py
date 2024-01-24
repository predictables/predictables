from typing import List, Optional, Union

import pandas as pd

from predictables.util import get_unique


def _get_data(
    df: pd.DataFrame,
    df_val: Optional[pd.DataFrame],
    element: str = "x",
    data: str = "train",
    fold_n: int = None,
    feature_col_name: str = "feature",
    target_col_name: str = "target",
    fold_col_name: str = "fold",
) -> List[Union[int, float, str]]:
    """
    Helper function to get the requested data element.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to get the data from.
    element : str, optional
        What data element to get. Choices are "x", "y", or "fold"
        for X data (features), y data (target), or data from the nth
        cv fold. Note that `n` must be a named cv fold in the data,
        or an error will be raised.
    data : str, optional
        What data to use for the plot. Choices are "train", "test",
        "all".
    fold_n : int, optional
        If element is "fold", which fold to get. Must be a named
        cv fold in the data.
    feature_col_name : str, optional
        The name of the feature column. The default is "feature".
    target_col_name : str, optional
        The name of the target column. The default is "target".
    fold_col_name : str, optional
        The name of the fold column. The default is "fold".

    Returns
    -------
    List[Union[int, float, str]]
        The values for the requested column.
    """
    element = element.lower()
    data = data.lower()
    unique_folds = get_unique(df[fold_col_name])

    if data not in ["train", "test", "all"]:
        raise ValueError(f"data must be one of 'train', 'test', or 'all'. Got {data}.")
    if element not in ["x", "y", "fold"]:
        raise ValueError(f"element must be one of 'x', 'y', or 'fold'. Got {element}.")
    if element == "fold" and fold_n not in unique_folds:
        raise ValueError(f"fold_n must be one of {unique_folds}. Got {fold_n}.")

    # Use the cv function if we're getting a fold
    if element == "fold":
        return _filter_df_for_cv(df, fold_n, fold_col_name, data)[
            feature_col_name
        ].tolist()

    # Otherwise, get the data for the requested fold
    return _filter_df_for_train_test(df, df_val, data)[
        feature_col_name if element == "x" else target_col_name
    ]


def _filter_df_for_cv(
    df: pd.DataFrame, fold: int, fold_col: str, data: str = "all"
) -> pd.DataFrame:
    """
    Get the data for the requested fold. This means that we only return
    rows having the cv label of the fold.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to get the data from.
    fold : int
        The fold number to get the data for. Must be a named
        cv fold in the data, or an error will be raised.
    fold_col : str
        The name of the fold column.
    data : str, optional
        What data to use for the plot. Choices are "train", "test", "all".

    Returns
    -------
    pd.DataFrame
        The data for the requested fold.

    Raises
    ------
    KeyError
        If fold is not a named cv fold in the data.
    KeyError
        If fold_col is not a column in the data.
    KeyError
        If data is not one of "train", "test", or "all".
    """
    if data not in ["train", "test", "all"]:
        raise ValueError(f"data must be one of 'train', 'test', or 'all'. Got {data}.")

    if fold_col not in df.columns:
        raise KeyError(f"No column in the DataFrame called '{fold_col}'")

    if fold not in get_unique(df[fold_col]):
        raise KeyError(f"{fold} is not a named cv fold in the DataFrame.")

    if data == "all":
        return df
    else:
        return (
            df.loc[df[fold_col].ne(fold), :]
            if data == "train"
            else df.loc[df[fold_col].eq(fold), :]
        )


def _filter_df_for_cv_train(df: pd.DataFrame, fold: int, fold_col: str) -> pd.DataFrame:
    """
    Get the training data for the requested fold. This means that we exclude
    all rows having the label of the fold, and return the remaining rows.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to get the data from.
    fold : int
        The fold number to get the training data for. Must be a named
        cv fold in the data, or an error will be raised.
    fold_col : str
        The name of the fold column.

    Returns
    -------
    pd.DataFrame
        The training data for the requested fold.

    Raises
    ------
    KeyError
        If fold is not a named cv fold in the data.
    KeyError
        If fold_col is not a column in the data.
    """
    if fold_col not in df.columns:
        raise KeyError(f"No column in the DataFrame called '{fold_col}'")

    if fold not in get_unique(df[fold_col]):
        raise KeyError(f"{fold} is not a named cv fold in the DataFrame.")

    return df.loc[df[fold_col].ne(fold), :]


def _filter_df_for_cv_test(df: pd.DataFrame, fold: int, fold_col: str) -> pd.DataFrame:
    """
    Get the testing/validation data for the requested fold. This means that
    we only return rows having the cv label of the fold.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to get the data from.
    fold : int
        The fold number to get the validation data for. Must be a named
        cv fold in the data, or an error will be raised.
    fold_col : str
        The name of the fold column.

    Returns
    -------
    pd.DataFrame
        The validation data for the requested fold.

    Raises
    ------
    KeyError
        If fold is not a named cv fold in the data.
    KeyError
        If fold_col is not a column in the data.
    """
    if fold_col not in df.columns:
        raise KeyError(f"No column in the DataFrame called '{fold_col}'")

    if fold not in get_unique(df[fold_col]):
        raise KeyError(f"{fold} is not a named cv fold in the DataFrame.")

    return df.loc[df[fold_col].eq(fold), :]


def _filter_df_for_train_test(
    df: pd.DataFrame, df_val: Optional[pd.DataFrame], data: str = "all"
) -> pd.DataFrame:
    """
    Returns a dataframe representing the `data` string input -- one of:
        `all` - returns the concatenated [df, df_val] set
        `train` - returns df
        `test` - returns df_val

    Parameters
    ----------
    df : pd.DataFrame
        The training dataframe.
    df_val : pd.DataFrame | None
        The validation dataframe.
    data : str, optional
        The data to return. The default is "all".

    Returns
    -------
    pd.DataFrame
        The requested dataframe.
    """
    data = data.lower()
    if data not in ["all", "train", "test"]:
        raise ValueError(
            f'"data" must be one of ["all", "train", "test"]. Got "{data}".'
        )

    if data == "all":
        return (
            pd.concat([df, df_val.assign(fold_col=-42)], axis=0)
            if df_val is not None
            else df
        )
    elif data == "train":
        return df
    elif data == "test":
        return df_val
    else:
        raise ValueError(f"data must be one of 'all', 'train', or 'test'. Got {data}.")
