"""
This module is designed to evaluate a model-based imputation method and define
the stopping criterion for an iterative training process. The key steps include:

1. Train the model iteratively over multiple epochs, using training data for
   learning and validation data for performance evaluation.
2. At each epoch, apply cross-validation to calculate the error metric (e.g.,
   mean squared error) for each fold.
3. Store the error metrics for each epoch across all folds.
4. After a sufficient number of epochs, calculate the standard error of the
   mean (SEM) of the error metrics starting from a specific epoch.
5. Use this SEM to establish a boundary (mean ± SEM) for assessing error convergence.
6. Check if the error metrics for the last two epochs are within this boundary.
   If they are, it indicates that the errors have converged, and the training
   process can be stopped.
7. Otherwise, continue the training process until the maximum number of epochs
   is reached or until the error metrics converge.

Note: The stopping criterion is based on the assumption that if the error in
      two consecutive epochs falls within one standard deviation of the mean
      error of prior epochs, it suggests convergence in the model's performance,
      warranting the cessation of further training.
"""

from typing import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold


def evaluate_imputation_one_column(
    imputed_df: pd.DataFrame,
    original_df: pd.DataFrame,
    missing_mask: pd.DataFrame,
    column: str,
) -> tuple:
    """
    Evaluate the imputation of a single column.

    :param imputed_df: The imputed dataframe.
    :param original_df: The original dataframe.
    :param missing_mask: A mask indicating where the original dataframe had missing values.
    :param column: The name of the column to evaluate.
    :return: The imputation error for the column.
    """
    # Check if the column exists in both dataframes
    if column not in imputed_df.columns or column not in original_df.columns:
        raise ValueError(f"Column '{column}' not found in both dataframes.")

    # Get the original and imputed values in rows without missing values
    original_values = original_df.loc[~missing_mask[column], column]
    imputed_values = imputed_df.loc[~missing_mask[column], column]

    # Calculate the mean absolute error (MAE)
    mae = abs(original_values - imputed_values).mean()

    # Handle division by zero in MAPE calculation
    nonzero_original_values = original_values[original_values != 0]
    mape = (
        (
            abs(
                nonzero_original_values
                - imputed_values.loc[nonzero_original_values.index]
            )
            / nonzero_original_values
        ).mean()
        if len(nonzero_original_values) > 0
        else None
    )

    return mae, mape


def train_model_on_fold(
    model, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray
) -> np.ndarray:
    """
    Train the model on training data and evaluate on validation data.

    :param model: The model to train.
    :param X_train: The training data.
    :param y_train: The training labels.
    :param X_val: The validation data.
    :return: The predictions on the validation data.
    """
    # Train the model
    model.fit(X_train, y_train)

    # Predict on validation set
    predictions = model.predict(X_val)

    return predictions


def calculate_fold_error(
    y_val: np.ndarray,
    predictions: np.ndarray,
    error_metric: Callable = mean_squared_error,
) -> float:
    """
    Calculate the error for a fold using the provided error metric.

    :param y_val: The validation labels.
    :param predictions: The predictions on the validation data.
    :param error_metric: The error metric to use.
    :return: The error for the fold.
    """
    return error_metric(y_val, predictions)


def cross_validate_model(
    model: Callable,
    X: np.ndarray,
    y: np.ndarray,
    n_folds: int,
    n_epochs: int,  # Assuming you have multiple epochs per fold
    error_metric: Callable = mean_squared_error,
) -> list:
    """
    Perform cross-validation and return the errors for each epoch in each fold.

    :param model: The model to train.
    :param X: The features.
    :param y: The labels.
    :param n_folds: The number of folds to use.
    :param n_epochs: The number of epochs to train each fold.
    :param error_metric: The error metric to use.
    :return: A list of lists containing errors for each epoch in each fold.
    """
    kf = KFold(n_splits=n_folds)
    fold_errors = []

    for train_index, val_index in kf.split(X):
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

        epoch_errors = []
        for epoch in range(n_epochs):
            model.fit(X_train, y_train)  # Assuming model retains state across epochs
            predictions = model.predict(X_val)
            error = error_metric(y_val, predictions)
            epoch_errors.append(error)

        fold_errors.append(epoch_errors)

    return fold_errors


def calculate_standard_error_of_mean(errors: list, start_epoch: int) -> float:
    """
    Calculate the standard error of the mean from a starting epoch to the last epoch.

    :param errors: A list of lists, each inner list containing errors for each epoch in a fold.
    :param start_epoch: The starting epoch to calculate the SEM from.
    :return: The standard error of the mean.
    """
    if not errors or not errors[0] or start_epoch >= len(errors[0]):
        raise ValueError("Invalid input for errors list or start_epoch.")

    # Extracting errors from the starting epoch to the last for all folds
    relevant_errors = np.array([fold_errors[start_epoch:] for fold_errors in errors]).T
    print(f"relevant_errors: {relevant_errors}")

    # Flatten the list and calculate SEM
    std_dev = np.array([np.std(x) for x in relevant_errors])
    print(f"std_dev: {std_dev}")
    sem_ = std_dev / np.sqrt(len(relevant_errors[0]))

    return sem_[0]


def check_stopping_criterion(errors):
    """
    The model will stop training if the error in two consecutive epochs is within one standard error of each other.
    """
    # Ensure there are at least three folds to compare
    if len(errors) < 3:
        return False

    # Latest epoch index
    latest_epoch = len(errors[0]) - 1

    # Calculate the standard error of the mean for the latest epoch
    sem_latest_epoch = calculate_standard_error_of_mean(errors, latest_epoch)

    # Compare the last two sets of errors
    upper_bound = errors[-3][latest_epoch] + sem_latest_epoch
    lower_bound = errors[-3][latest_epoch] - sem_latest_epoch

    return (lower_bound <= errors[-2][latest_epoch] <= upper_bound) and (
        lower_bound <= errors[-1][latest_epoch] <= upper_bound
    )
