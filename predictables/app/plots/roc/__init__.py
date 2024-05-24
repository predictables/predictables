from __future__ import annotations

import streamlit as st
import pandas as pd
import numpy as np

from predictables.app.plots.roc.src.data_preparation import prepare_roc_data
from predictables.app.plots.roc.src.roc_auc_calculation import (
    calculate_mean_roc_auc,
    calculate_roc_auc,
)
from predictables.app.plots.roc.src.plot_generation import generate_roc_auc_plot


def roc_curve(
    df: pd.DataFrame, univariate_feature_variable: str, target_variable: str
) -> None:
    """Generate ROC curve for the univariate analysis."""
    ## ROC Curve
    roc_data = prepare_roc_data(df, use_time_series_validation=True)

    # Calculate ROC AUC for each fold
    roc_curves = []
    features = [univariate_feature_variable]
    target = target_variable

    for train_data, validation_data in roc_data:
        roc_curve = calculate_roc_auc(train_data, validation_data, features, target)
        roc_curves.append(roc_curve)

    # Calculate mean ROC AUC and standard error
    mean_fpr, mean_tpr, std_error = calculate_mean_roc_auc(roc_curves)

    # Generate ROC AUC plot
    p = generate_roc_auc_plot(roc_curves, mean_fpr, mean_tpr, std_error)

    df = pd.DataFrame(
        {
            "Fold": [f"Fold-{i}" for i in range(1, len(roc_curves) + 1)]
            + ["Mean"]
            + ["Std Error"],
            "ROC AUC": [roc_auc for _, _, roc_auc in roc_curves]
            + [np.mean([roc_auc for _, _, roc_auc in roc_curves])]
            + [
                np.std([roc_auc for _, _, roc_auc in roc_curves])
                / np.sqrt(len(roc_curves))
            ],
        }
    ).set_index("Fold")

    df["ROC AUC"] = df["ROC AUC"].apply(lambda x: f"{x:.1%}")

    return p, df

def roc_curve_general(
    df: pd.DataFrame, feature_variables: str | list[str], target_variable: str
) -> None:
    """Generate ROC curve for the univariate analysis."""
    ## ROC Curve
    roc_data = prepare_roc_data(df, use_time_series_validation=True)

    # Calculate ROC AUC for each fold
    roc_curves = []
    features = (
        [feature_variables] if isinstance(feature_variables, str) else feature_variables
    )
    target = target_variable

    for train_data, validation_data in roc_data:
        roc_curve = calculate_roc_auc(train_data, validation_data, features, target)
        roc_curves.append(roc_curve)

    # Calculate mean ROC AUC and standard error
    mean_fpr, mean_tpr, std_error = calculate_mean_roc_auc(roc_curves)

    # Generate ROC AUC plot
    p = generate_roc_auc_plot(roc_curves, mean_fpr, mean_tpr, std_error)

    df = pd.DataFrame(
        {
            "Fold": [f"Fold-{i}" for i in range(1, len(roc_curves) + 1)]
            + ["Mean"]
            + ["Std Error"],
            "ROC AUC": [roc_auc for _, _, roc_auc in roc_curves]
            + [np.mean([roc_auc for _, _, roc_auc in roc_curves])]
            + [
                np.std([roc_auc for _, _, roc_auc in roc_curves])
                / np.sqrt(len(roc_curves))
            ],
        }
    ).set_index("Fold")

    df["ROC AUC"] = df["ROC AUC"].apply(lambda x: f"{x:.1%}")

    return p, df
