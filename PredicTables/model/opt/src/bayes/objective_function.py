import pandas as pd
import numpy as np
import polars as pl
from sklearn.base import BaseEstimator
from typing import Union
from PredicTables.util import to_pd_df, to_pd_s


def _objective_function_no_pruning(params):
    """
    This is for testing purposes only. It is not used in the actual
    optimization process. It is a simple objective function that seeks
    to minimize the L1 norm of the parameters.

    Parameters
    ----------
    params : dict
        Hyperparameters for the model.

    Returns
    -------
    float
        The sum of the absolute values of the parameters.
    """
    # Absolute value of the parameters
    abs_params = np.abs(list(params.values()))

    # Return the sum of the absolute values of the parameters
    return np.sum(abs_params)


def objective_with_pruning(
    params: dict,
    model: BaseEstimator,
    X_train: Union[pd.DataFrame, pl.DataFrame, pl.LazyFrame],
    y_train: Union[pd.Series, pl.Series],
    X_val: Union[pd.DataFrame, pl.DataFrame, pl.LazyFrame],
    y_val: Union[pd.Series, pl.Series],
    pruning_threshold: float,
    n_checkpoints: int = 10,
) -> float:
    """
    Objective function that includes logic for pruning.

    Parameters
    ----------
    params : dict
        Hyperparameters for the model.
    model : sklearn.base.BaseEstimator
        Machine learning model that supports partial fitting or early stopping,
        typically inheriting from sklearn.base.BaseEstimator.
    X_train, y_train : array-like
        Training features and labels.
    X_val, y_val : array-like
        Validation features and labels. Used for pruning.
    pruning_threshold : float
        Threshold for pruning. If the model's performance on the validation set
        is worse than this threshold, the model is pruned.

    Returns
    -------
    float
        The model's performance on the validation set.
    """
    # Convert to pandas
    X_train = to_pd_df(X_train)
    y_train = to_pd_s(y_train)
    X_val = to_pd_df(X_val)
    y_val = to_pd_s(y_val)

    # Initialize the best score to infinity
    best_evaluation = float("inf")

    # Loop over checkpoints, fitting the model at each checkpoint, evaluating,
    # and making sure the improvement justifies additional training time
    # (i.e., pruning)
    pct_to_train = 1.0 / n_checkpoints
    for checkpoint in range(1, n_checkpoints + 1):
        # Calculate the number of iterations to train for at this checkpoint
        pct_trained = pct_to_train * checkpoint
        n_iterations = int(pct_trained * params["n_iterations"])

        # Update the model with the new number of iterations
        model = partial_fit(X_train, y_train, model, n_iterations=n_iterations)

        # Evaluate the model
        current_evaluation = evaluate_partial_fit(X_val, y_val, model)

        # Pruning condition: if current score is worse than the threshold, prune
        if should_model_be_pruned(
            current_evaluation, best_evaluation, pruning_threshold
        ):
            break

        # If not pruned, update the best score
        best_evaluation = current_evaluation

    # By exiting the loop, we have either pruned the model or trained it to
    # completion. If the model was pruned, we need to retrain it to completion
    # before returning the final evaluation.
    model = complete_fit(X_train, y_train, model, **params)

    # Evaluate the model
    final_evaluation = evaluate_model(X_val, y_val, model)

    # Return the best evaluation if the model was pruned, otherwise return
    # the final evaluation
    final_score = (
        best_evaluation if best_evaluation < final_evaluation else final_evaluation
    )

    return final_score


def partial_fit(X_train, y_train, model):
    # Partially fit the model here
    # Depending on the model, different approaches may be needed
    # such as early stopping, training for a certain number of iterations, etc.
    pass


def evaluate_partial_fit(X_val, y_val, model):
    # Evaluate the model here
    # Evaluate consistently with ultimate goal of optimization process --
    # e.g., if optimizing for accuracy, evaluate using accuracy, not AUC
    pass


def should_model_be_pruned(new_score, current_score, pruning_factor=1, cond="gt"):
    # Define pruning condition here
    # implementing a simple one for now
    if cond == "gt":
        return current_score > (pruning_factor * new_score)
    elif cond == "lt":
        return current_score < (pruning_factor * new_score)
    else:
        raise NotImplementedError(
            f"Condition {cond} not implemented. Currently only 'gt' and 'lt' are supported."
        )


def complete_fit(X_train, y_train, model, **params):
    # Fit the model here
    pass


def evaluate_model(X_val, y_val, model):
    # Evaluate the model here
    # Evaluate consistently with ultimate goal of optimization process --
    # e.g., if optimizing for accuracy, evaluate using accuracy, not AUC
    pass
