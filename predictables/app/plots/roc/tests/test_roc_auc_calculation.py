import pytest
import pandas as pd
import numpy as np
from predictables.app.plots.roc.src.roc_auc_calculation import (
    calculate_roc_auc,
    calculate_mean_roc_auc,
)


def test_calculate_roc_auc():
    data = pd.DataFrame(
        {
            "feature": [1, 2, 3, 4, 5, 6],
            "target": [0, 1, 0, 1, 0, 1],
            "fold": [0, 0, 1, 1, 2, 2],
        }
    )
    features = ["feature"]
    target = "target"
    train_data = data[data["fold"] != 1]
    validation_data = data[data["fold"] == 1]
    fpr, tpr, roc_auc = calculate_roc_auc(train_data, validation_data, features, target)
    assert isinstance(fpr, np.ndarray), f"Expected np.ndarray, got {type(fpr)}"
    assert isinstance(tpr, np.ndarray), f"Expected np.ndarray, got {type(tpr)}"
    assert isinstance(roc_auc, float), f"Expected float, got {type(roc_auc)}"


def test_calculate_mean_roc_auc():
    roc_curves = [
        (np.array([0.0, 0.5, 1.0]), np.array([0.0, 0.8, 1.0]), 0.9),
        (np.array([0.0, 0.4, 1.0]), np.array([0.0, 0.7, 1.0]), 0.85),
    ]
    mean_fpr, mean_tpr, std_error = calculate_mean_roc_auc(roc_curves)
    assert isinstance(
        mean_fpr, np.ndarray
    ), f"Expected np.ndarray, got {type(mean_fpr)}"
    assert isinstance(
        mean_tpr, np.ndarray
    ), f"Expected np.ndarray, got {type(mean_tpr)}"
    assert isinstance(
        std_error, np.ndarray
    ), f"Expected np.ndarray, got {type(std_error)}"
