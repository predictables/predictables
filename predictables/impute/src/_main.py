from typing import Union

import pandas as pd
import polars as pl

from predictables.impute.src._get_cv_folds import get_cv_folds
from predictables.impute.src._get_missing_data_mask import (
    get_missing_data_mask,
)
from predictables.impute.src._impute_with_trained_model import (
    impute_with_trained_model,
)
from predictables.impute.src._initial_impute import initial_impute
from predictables.impute.src._train_catboost_model import train_catboost_model


def integrated_imputation_workflow(
    df: Union[pd.DataFrame, pl.DataFrame, pl.LazyFrame],
    learning_rate: float = 0.1,
):
    """
    This function represents the integrated workflow for imputing missing values using CatBoost.

    Parameters
    ----------
    df : Union[pd.DataFrame, pl.DataFrame, pl.LazyFrame]
        The df to impute missing values for.
    learning_rate : float, optional
        The learning rate for the CatBoost model, by default 0.1.

    Returns
    -------
    pl.LazyFrame
        The imputed df.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
    >>> df.iloc[0, 0] = None
    >>> df.iloc[1, 1] = None
    >>> df.iloc[2, 1] = None
    >>> df
         a    b
    0  NaN  4.0
    1  2.0  NaN
    2  3.0  NaN
    >>> from predictables.impute.src.main import integrated_imputation_workflow
    >>> integrated_imputation_workflow(df)
         a         b
    0  2.0  4.000000
    1  2.0  4.666667
    2  3.0  5.333333
    """
    # Get a missing mask to store which values are missing (and are therefore imputed)
    missing_mask = get_missing_data_mask(df)

    # Initial Imputation - will impute missing values with the median
    # or mode depending on the column type
    df_imputed = initial_impute(df)

    # Generating Cross-Validation Folds
    cv_folds = get_cv_folds(df_imputed)

    # Model Training
    trained_models = train_catboost_model(df_imputed, cv_folds)

    # Imputation with the Trained Models
    final_df = impute_with_trained_model(
        df_imputed,
        missing_mask,
        trained_models,
        learning_rate,
        only_missing=True,
    )

    return final_df
