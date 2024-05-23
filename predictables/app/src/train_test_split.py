"""Split the data into training and testing sets."""

from __future__ import annotations
import pandas as pd


def train_test_split(
    X: pd.Series | pd.DataFrame,
    y: pd.Series,
    folds: pd.Series,
    fold: int,
    time_series_validation: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split the data into training and testing sets, depending on the validation type."""
    if time_series_validation:
        return time_series_train_test_split(X, y, folds, fold)

    return standard_train_test_split(X, y, folds, fold)


def time_series_train_test_split(
    X: pd.Series | pd.DataFrame, y: pd.Series, folds: pd.Series, fold: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split the data into training and testing sets."""
    x_train = X[folds < fold] if isinstance(X, pd.Series) else X.loc[folds < fold]
    x_test = X[folds == fold] if isinstance(X, pd.Series) else X.loc[folds == fold]

    y_train = y[folds < fold]
    y_test = y[folds == fold]

    return x_train, x_test, y_train, y_test


def standard_train_test_split(
    X: pd.Series | pd.DataFrame, y: pd.Series, folds: pd.Series, fold: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split the data into training and testing sets."""
    x_train = X[folds != fold] if isinstance(X, pd.Series) else X.loc[folds != fold]
    x_test = X[folds == fold] if isinstance(X, pd.Series) else X.loc[folds == fold]

    y_train = y[folds != fold]
    y_test = y[folds == fold]

    return x_train, x_test, y_train, y_test