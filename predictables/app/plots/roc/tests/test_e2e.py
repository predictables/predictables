import pytest
import pandas as pd
import numpy as np
from predictables.app.plots.roc.src.data_preparation import prepare_roc_data
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


def test_e2e(sample_data):
    # Load and prepare data
    use_time_series_validation = True  # Or set based on user input
    prepared_data = prepare_roc_data(sample_data, use_time_series_validation)

    # Calculate ROC AUC for each fold
    roc_curves = []
    features = ["feature"]  # Replace with actual feature columns
    target = "target"

    for train_data, validation_data in prepared_data:
        roc_curve = calculate_roc_auc(train_data, validation_data, features, target)
        roc_curves.append(roc_curve)

    # Calculate mean ROC AUC and standard error
    mean_fpr, mean_tpr, std_error = calculate_mean_roc_auc(roc_curves)

    # Generate ROC AUC plot
    generate_roc_auc_plot(roc_curves, mean_fpr, mean_tpr, std_error)
