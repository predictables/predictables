from concurrent.futures import ProcessPoolExecutor
from typing import Tuple

import pandas as pd

from predictables.impute.src.evaluate_imputation import check_stopping_criterion
from predictables.impute.src.get_missing_data_mask import get_missing_data_mask
from predictables.impute.src.impute_with_trained_model import impute_with_trained_model
from predictables.impute.src.initial_impute import initial_impute
from predictables.impute.src.train_catboost_model import train_one_catboost_model


def train_and_predict(
    df: pd.DataFrame, missing_mask: pd.DataFrame, column: str
) -> Tuple[str, pd.Series]:
    """
    Train a model for a given column and predict missing values.
    """
    # Initialize and fit the model (CatBoost or any other model)
    model = train_one_catboost_model(df, column)

    # Predict missing values
    predictions = impute_with_trained_model(df, missing_mask, model)

    return column, predictions


def unpack_and_train_predict(args):
    """
    Unpacks arguments and calls train_and_predict.
    """
    return train_and_predict(*args)


def iterative_imputation_with_multiprocessing(
    df: pd.DataFrame, max_iterations: int = 24, max_workers: int = 8
) -> pd.DataFrame:
    """
    Perform iterative imputation using multiprocessing.

    :param df: DataFrame with missing values to impute.
    :param max_iterations: Maximum number of iterations for the imputation process.
    :param max_workers: Maximum number of worker processes.
    :return: DataFrame with imputed values.
    """
    missing_mask = get_missing_data_mask(df)
    df_imputed = initial_impute(df)

    df = df.collect().to_pandas()
    df_imputed = df_imputed.collect().to_pandas()
    missing_mask = missing_mask.collect().to_pandas()
    for _ in range(max_iterations):
        training_data = [
            (df_imputed, missing_mask, col)
            for col in df.columns
            if missing_mask[col].any()
        ]

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Use a lambda function to unpack and pass arguments
            results = executor.map(unpack_and_train_predict, training_data)

        # Update the dataframe with predictions from each model
        for column, col_predictions in results:
            df_imputed.loc[missing_mask[column], column] = col_predictions

        # Check for convergence using your defined stopping criteria
        if check_stopping_criterion(df, df_imputed):
            break

    return df_imputed
