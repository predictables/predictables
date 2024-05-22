"""Prepare the data for the ROC-AUC plot."""

from __future__ import annotations
import pandas as pd


def load_data(filepath: str, fold_column: str = "fold") -> pd.DataFrame:
    """Load the data from a CSV file.

    Parameters
    ----------
    filepath : str
        Path to the CSV file.

    Returns
    -------
    DataFrame
        Loaded data.
    """
    df = pd.read_csv(filepath)

    if fold_column not in df.columns:
        raise KeyError(f"Column '{fold_column}' not found in the data:\n{df.head()}")

    return df


def prepare_roc_data(
    data: pd.DataFrame, use_time_series_validation: bool
) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """Prepare the data based on the use_time_series_validation flag.

    Parameters
    ----------
    data : DataFrame
        The input data.
    use_time_series_validation : bool
        Flag to determine the type of cross-validation.

    Returns
    -------
    list of tuples
        List containing training and validation sets for each fold.
    """
    folds = data["fold"].unique()
    prepared_data = []

    if use_time_series_validation:
        for fold in folds:
            train_data = data[data["fold"] <= fold]
            validation_data = data[data["fold"] == fold + 1]
            if not validation_data.empty:
                prepared_data.append((train_data, validation_data))
    else:
        for fold in folds:
            train_data = data[data["fold"] != fold]
            validation_data = data[data["fold"] == fold]
            if not validation_data.empty:
                prepared_data.append((train_data, validation_data))

    return prepared_data
