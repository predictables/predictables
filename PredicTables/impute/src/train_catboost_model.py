from typing import Union

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor

from PredicTables.util import to_pd_df

from .get_cv_folds import get_cv_folds


def train_one_catboost_model(
    df: pd.DataFrame, target_column: str, cv_folds: Union[int, list] = None
) -> Union[CatBoostRegressor, CatBoostClassifier]:
    """
    Trains a CatBoost model (regressor or classifier) based on the target column's data type.

    :param df: The df to train the model on. Should contain the features and target column.
    :type df: pd.DataFrame
    :param target_column: The name of the target column.
    :type target_column: str
    :param cv_folds: The number of cross-validation folds to use. If None, will not do cross validation. If an integer is provided, will create that many folds. If a list is provided and is the same length as the df, will use those indices to create the folds. If the size if not the same as the df, will raise an error.
    :type cv_folds: Union[int, list]
    :return: A trained CatBoost model.
    :rtype: Union[CatBoostRegressor, CatBoostClassifier]
    """
    df = to_pd_df(df)
    # Check inputs
    assert isinstance(
        df, pd.DataFrame
    ), f"df must be a pandas DataFrame, not {type(df)}"
    assert isinstance(
        target_column, str
    ), f"target_column must be a string, not {type(target_column)}"
    assert target_column in df.columns, f"{target_column} is not in df.columns"
    assert isinstance(cv_folds, (int, list)) or (
        cv_folds is None
    ), f"cv_folds must be an integer or list, not {type(cv_folds)}"
    assert (
        (cv_folds is None) or (isinstance(cv_folds, int) or len(cv_folds) == len(df))
    ), f"cv_folds must be an integer or a list of the same length as df. len(cv_folds): {len(cv_folds)}, len(df): {len(df)}"
    assert (cv_folds is None) or (
        isinstance(cv_folds, int) or all(isinstance(i, int) for i in cv_folds)
    ), f"cv_folds must be an integer or a list of integers. cv_folds: {cv_folds}"

    # Get the cross-validation folds (if cv_folds is not None)
    if cv_folds is not None:
        if isinstance(cv_folds, int):
            cv_folds = get_cv_folds(df, cv_folds)
        elif isinstance(cv_folds, list):
            cv_folds = cv_folds

    # Separating features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # If cv_folds is not None, train a model for each fold
    if cv_folds is not None:
        # Initialize an empty list to store the trained models
        trained_models = []

        # Loop through each fold
        for i, (train_idx, test_idx) in enumerate(cv_folds):
            # Train a model for the fold - regressor if the target is numeric, classifier if the target is categorical
            try:
                if y.dtype in [np.float64, np.int64]:
                    model = CatBoostRegressor()
                else:
                    model = CatBoostClassifier()
            except Exception as _:
                # print(f"Warning: {e} - trying to convert y to string")
                y = y.astype(str)
                model = CatBoostClassifier()

            # Get categorical features
            categorical_features = X.select_dtypes(include="category").columns.tolist()

            # Fit the model
            model.fit(
                X.iloc[train_idx],
                y.iloc[train_idx],
                verbose=False,
                cat_features=categorical_features,
            )

            # Add the trained model to the list
            trained_models.append(
                {
                    "fold": i,
                    "model": model,
                    "train_idx": train_idx,
                    "test_idx": test_idx,
                }
            )

        return trained_models

    else:
        # Train a model - regressor if the target is numeric, classifier if the target is categorical
        try:
            if np.issubdtype(y.dtype, np.number):
                model = CatBoostRegressor()
            else:
                model = CatBoostClassifier()
        except Exception as _:
            # print(f"Warning: {e} - trying to convert y to string")
            y = y.astype(str)
            model = CatBoostClassifier()

        # Get categorical features
        categorical_features = X.select_dtypes(include="category").columns.tolist()

        # Fit the model
        model.fit(X, y, verbose=False, cat_features=categorical_features)

        return model


def train_catboost_model(df, missing_mask, cv_folds: Union[int, list] = None):
    """
    Trains a CatBoost model (regressor or classifier) for each column in the df.

    :param df: The df to train the models on.
    :param missing_mask: The mask indicating which values are missing in the df.
    :return: A dictionary of trained CatBoost models.
    """
    df = to_pd_df(df)
    missing_mask = to_pd_df(missing_mask)
    # Check inputs
    assert isinstance(
        df, pd.DataFrame
    ), f"df must be a pandas DataFrame, not {type(df)}"
    assert isinstance(
        missing_mask, pd.DataFrame
    ), f"missing_mask must be a pandas DataFrame, not {type(missing_mask)}"
    assert (
        df.shape == missing_mask.shape
    ), f"df and missing_mask must have the same shape. df.shape: {df.shape}, missing_mask.shape: {missing_mask.shape}"
    # Initialize an empty dictionary to store the trained models
    trained_models = {}

    # Loop through each column in the df
    for column in df.columns:
        # Skip the column if it has no missing values
        if missing_mask[column].any():
            # Train a model for the column if it has missing values
            trained_models[column] = [train_one_catboost_model(df, column, cv_folds)]

    return trained_models
