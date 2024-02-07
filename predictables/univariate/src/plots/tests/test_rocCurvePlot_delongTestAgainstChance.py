import pandas as pd  # type: ignore
import numpy as np
from typing import Tuple
import logging
import pytest
from sklearn.metrics import roc_auc_score  # type: ignore

from predictables.univariate.src.plots._roc_curve_plot import (
    _delong_test_against_chance,
)
from predictables.util import get_unique
import os
from dotenv import load_dotenv

load_dotenv()
log_level = os.getenv("LOGGING_LEVEL")


@pytest.fixture
def cancer():
    """Load the breast cancer dataset."""
    return pd.read_parquet("cancerdf.parquet")


@pytest.fixture
def sample_data() -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Generate sample data for testing."""
    np.random.seed(42)  # Ensure reproducibility
    yhat_proba_logits = pd.Series(np.random.rand(100))
    yhat_proba = 1 / (1 + np.exp(-yhat_proba_logits))  # Convert to probabilities
    y = pd.Series(
        np.random.binomial(1, yhat_proba, size=100)
    )  # Simulate binary outcomes
    fold = pd.Series(np.random.choice([1, 2, 3, 4, 5], size=100))
    return y, yhat_proba, fold


@pytest.fixture
def sample_variance(sample_data: Tuple[pd.Series, pd.Series, pd.Series]) -> float:
    """Calculate the variance of the AUC for the sample data."""
    y, yhat_proba, fold = sample_data
    aucs = [
        roc_auc_score(y[fold == f], yhat_proba[fold == f]) for f in get_unique(fold)
    ]
    return float(np.var(aucs))


@pytest.fixture
def sample_variance_bootstrap(
    sample_data: Tuple[pd.Series, pd.Series, pd.Series],
):
    y, yhat_proba, _ = sample_data
    idx = np.array(
        [np.random.choice(len(y), len(y), replace=True) for _ in range(2500)]
    )

    # Turn y and yhat from a pd.Series (vector) to a np.array (matrix) of shape (n, 2500)
    y = np.array([y.iloc[i] for i in idx])
    yhat_proba = np.array([yhat_proba.iloc[i] for i in idx])

    # Calculate the AUC for each bootstrap sample
    aucs = np.array([roc_auc_score(y[i], yhat_proba[i]) for i in range(2500)])

    # Calculate the mean and standard deviation of the AUCs
    logging.info(f"Mean AUC: {np.mean(aucs)}, std AUC: {np.std(aucs)}")
    return np.mean(aucs), np.std(aucs)


@pytest.fixture
def large_sample_data():
    """Generate larger sample data for testing."""
    np.random.seed(42)
    size = 100000  # Increase the size for a more realistic test case
    yhat_proba_logits = pd.Series(np.random.rand(size))
    yhat_proba = 1 / (1 + np.exp(-yhat_proba_logits))
    y = pd.Series(np.random.binomial(1, yhat_proba, size=size))
    fold = pd.Series(np.random.choice([1, 2, 3, 4, 5], size=size))
    return y, yhat_proba, fold


def test_delong_test_against_chance_basic(sample_data):
    """Test basic functionality of the DeLong test."""
    y, yhat_proba, fold = sample_data
    z_stat, p_value = _delong_test_against_chance(y, yhat_proba, fold)
    assert (
        isinstance(z_stat, float) and isinstance(p_value, float)
    ), f"Z-statistic {z_stat} and p-value {p_value} should be floats, not {type(z_stat)} and {type(p_value)}."


def test_delong_test_against_chance_known_values(cancer):
    """Test the DeLong test against a the decomposed cancer dataset."""
    from sklearn.linear_model import LogisticRegression  # type: ignore
    from scipy.stats import norm  # type: ignore

    # Load the breast cancer data
    df = cancer
    X = df["comp_1"]
    y = df["target"]
    folds = df["cv"]

    # Fit a logistic regression model
    clf_total = LogisticRegression()
    clf_total.fit(X.values.reshape(-1, 1), y)

    # Predict probabilities
    yhat_proba = clf_total.predict_proba(X.values.reshape(-1, 1))[:, 1]

    # ROC AUC
    auc = roc_auc_score(y, yhat_proba)

    # Fit a logistic regression model for each fold, predict on validation set
    aucs = []
    for fold in get_unique(folds):
        idx_train = folds != fold
        idx_val = folds == fold
        clf = LogisticRegression()
        clf.fit(X[idx_train].values.reshape(-1, 1), y[idx_train])
        yhat_proba_val = clf.predict_proba(X[idx_val].values.reshape(-1, 1))[:, 1]
        auc_val = roc_auc_score(y[idx_val], yhat_proba_val)
        aucs.append(auc_val)

    # DeLong Z-statistic
    andy_Z = (auc - 0.5) / np.sqrt((auc * (1 - auc) + 0.25) / len(y))

    # DeLong p-value
    andy_p_value = 2 * (1 - norm.cdf(np.abs(andy_Z)))

    # DeLong test against chance
    z_stat, p_value = _delong_test_against_chance(y, yhat_proba, folds)

    # Test they are pretty close
    assert (
        np.abs(z_stat - andy_Z) < 5
    ), f"Z-statistic {z_stat} should be close to the one I calcluated: {andy_Z}."
    assert (
        np.abs(p_value - andy_p_value) < 0.1
    ), f"p-value {p_value} should be close to the one I calcluated: {andy_p_value}."


def test_delong_test_gives_significant_estimate(cancer):
    """Test the DeLong test with (almost) cancer dataset I know is significant."""
    from sklearn.linear_model import LogisticRegression  # type: ignore

    df = cancer
    X = df["comp_1"]
    y = df["target"]
    folds = df["cv"]

    # Fit a logistic regression model
    clf_total = LogisticRegression()
    clf_total.fit(X.values.reshape(-1, 1), y)

    # Predict probabilities
    yhat_proba = clf_total.predict_proba(X.values.reshape(-1, 1))[:, 1]

    _, p_value = _delong_test_against_chance(y, yhat_proba, folds)
    assert (
        p_value < 0.05
    ), "P-value should indicate significance for perfect prediction."


def test_delong_test_random_prediction():
    """Test the DeLong test with random predictions."""
    np.random.seed(42)
    y = pd.Series(np.random.binomial(1, 0.5, 10000))
    yhat_proba = pd.Series(np.random.rand(10000))
    fold = pd.Series(np.random.choice([1, 2, 3, 4, 5], 10000))

    z_stat, p_value = _delong_test_against_chance(y, yhat_proba, fold)
    assert (
        p_value > 0.05
    ), f"P-value ({p_value}) should not indicate significance for random predictions, but the z-score is {z_stat}."


@pytest.mark.parametrize("constant_value", [0, 0.5, 1])
def test_delong_test_constant_predictions(constant_value):
    """Test the DeLong test with constant predictions."""
    y = pd.Series([0, 1] * 50)
    yhat_proba = pd.Series([constant_value] * 100)
    fold = pd.Series([1, 2] * 50)

    with pytest.raises(ValueError):
        _delong_test_against_chance(y, yhat_proba, fold)


def test_delong_test_invalid_inputs():
    """Test the DeLong test with invalid inputs."""
    y = pd.Series([0, 1])
    yhat_proba = pd.Series([0.1, 0.9, 0.8])  # Mismatched length
    fold = pd.Series([1, 2])

    with pytest.raises(ValueError):
        _delong_test_against_chance(y, yhat_proba, fold)


@pytest.mark.parametrize("n_folds", [(5,), (10,), (20,)])
@pytest.mark.parametrize("p", [(0.25,), (0.5,), (0.75,)])
def test_delong_test_cross_validation_consistency(n_folds, p):
    """Test consistency of DeLong test results across cross-validation splits."""
    from sklearn.model_selection import KFold  # type: ignore

    np.random.seed(42)
    y = pd.Series(np.random.binomial(1, p[0], 10000))
    yhat_proba = pd.Series(np.random.rand(10000))
    kf = KFold(n_splits=n_folds[0], shuffle=True, random_state=42)

    z_stats = []
    p_values = []
    for _, test_index in kf.split(y):
        y_test = y.iloc[test_index].reset_index(drop=True)
        yhat_proba_test = yhat_proba.iloc[test_index].reset_index(drop=True)
        fold = pd.Series(test_index % n_folds[0]).reset_index(drop=True)
        z_stat, p_value = _delong_test_against_chance(y_test, yhat_proba_test, fold)
        z_stats.append(z_stat)
        p_values.append(p_value)

    # Assert the variance of Z-stats and p-values across folds is within reasonable bounds
    logging.debug(f"Variance of Z-stats: {np.var(z_stats)}")
    logging.debug(f"Variance of p-values: {np.var(p_values)}")
    assert np.var(z_stats) < 1, "Z-statistics across CV folds vary too much."
    assert np.var(p_values) < 0.1, "P-values across CV folds vary too much."
