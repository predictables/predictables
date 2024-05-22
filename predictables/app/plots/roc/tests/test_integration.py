"""Test the integration of the data preparation, ROC AUC calculation, and plot generation."""

import pytest
import pandas as pd
import numpy as np
from predictables.app.plots.roc.src.data_preparation import load_data, prepare_roc_data
from predictables.app.plots.roc.src.roc_auc_calculation import (
    calculate_roc_auc,
    calculate_mean_roc_auc,
)
from predictables.app.plots.roc.src.plot_generation import generate_roc_auc_plot
from sklearn.datasets import load_breast_cancer


@pytest.fixture
def sample_data():
    cancer = load_breast_cancer()
    df = pd.DataFrame(cancer.data, columns=cancer.feature_names)
    df["target"] = cancer.target
    df["fold"] = np.random.default_rng(42).integers(0, 5, df.shape[0])
    df.columns = df.columns.str.lower().str.replace(" ", "_")
    return pd.DataFrame(
        {"feature": df["mean_radius"], "target": df["target"], "fold": df["fold"]}
    )


def test_integration_time_series(sample_data):
    use_time_series_validation = True
    prepared_data = prepare_roc_data(sample_data, use_time_series_validation)

    roc_curves = []
    features = ["feature"]
    target = "target"

    for train_data, validation_data in prepared_data:
        roc_curve = calculate_roc_auc(train_data, validation_data, features, target)
        roc_curves.append(roc_curve)

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

    plot = generate_roc_auc_plot(roc_curves, mean_fpr, mean_tpr, std_error)
    assert (
        plot.title.text == "ROC AUC Plot"
    ), f"Expected 'ROC AUC Plot', got {plot.title.text}"


def test_integration_normal_cv(sample_data):
    use_time_series_validation = False
    prepared_data = prepare_roc_data(sample_data, use_time_series_validation)

    roc_curves = []
    features = ["feature"]
    target = "target"

    for train_data, validation_data in prepared_data:
        roc_curve = calculate_roc_auc(train_data, validation_data, features, target)
        roc_curves.append(roc_curve)

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

    plot = generate_roc_auc_plot(roc_curves, mean_fpr, mean_tpr, std_error)
    assert (
        plot.title.text == "ROC AUC Plot"
    ), f"Expected 'ROC AUC Plot', got {plot.title.text}"