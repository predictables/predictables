import pandas as pd
import numpy as np
import polars as pl
import sklearn

from PredicTables.util import to_pd_df, to_pd_s
from typing import Union, Tuple


def validate_input(
    X: Union[pd.DataFrame, pl.DataFrame, pl.LazyFrame],
    y: Union[pd.Series, pl.Series],
    model: sklearn.base.BaseEstimator,
) -> Tuple[bool, str]:
    """
    Validates the input dataset and model.

    Parameters
    ----------
    X : Union[pd.DataFrame, pl.DataFrame, pl.LazyFrame]
        Input dataset of model features.
    y : Union[pd.Series, pl.Series]
        Target variable.
    model : sklearn.base.BaseEstimator
        Model to be validated. Must be a scikit-learn estimator, or an estimator that
        implements the scikit-learn API, such as xgboost, lightgbm, etc.

    """
    # Check if the dataset can be converted to a pandas DataFrame
    try:
        X = to_pd_df(X)
    except Exception:
        return False, "Dataset cannot be converted to a pandas DataFrame."

    # Check if the target variable can be converted to a pandas Series
    try:
        y = to_pd_s(y)
    except Exception:
        return False, "Target variable cannot be converted to a pandas Series."

    # Convert to pandas DataFrame/series if necessary
    X = to_pd_df(X)
    y = to_pd_s(y)

    # Check if the dataset is a DataFrame
    if not isinstance(X, pd.DataFrame):
        return False, "Dataset is not a pandas DataFrame."

    # Check if the target variable is a Series
    if not isinstance(y, pd.Series):
        return False, "Target variable is not a pandas Series."

    # Check if the model has fit and predict methods
    if not (hasattr(model, "fit") and hasattr(model, "predict")):
        return False, "Model does not have 'fit' and 'predict' methods."

    return True, "Input validation successful."
