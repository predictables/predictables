from __future__ import annotations

import logging
from typing import Tuple

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score  # type: ignore

from predictables.univariate.src.plots._roc_curve_plot import _empirical_auc_variance
from predictables.util import get_unique


@pytest.fixture
def sample_data() -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Generate sample data for testing."""
    rg = np.random.default_rng(42)
    yhat_proba_logits = pd.Series(rg.random(100))
    yhat_proba = 1 / (1 + np.exp(-yhat_proba_logits))  # Convert to probabilities
    y = pd.Series(rg.binomial(1, yhat_proba, size=100))  # Simulate binary outcomes
    fold = pd.Series(rg.choice([1, 2, 3, 4, 5], size=100))
    return y, pd.Series(yhat_proba), fold


@pytest.fixture
def sample_variance(sample_data: Tuple[pd.Series, pd.Series, pd.Series]) -> float:
    """Calculate the variance of the AUC for the sample data."""
    y, yhat_proba, fold = sample_data
    aucs = [
        roc_auc_score(y[fold == f], yhat_proba[fold == f]) for f in get_unique(fold)
    ]
    return float(np.var(aucs))


@pytest.fixture
def sample_variance_bootstrap(sample_data: Tuple[pd.Series, pd.Series, pd.Series]):
    y, yhat_proba, _ = sample_data
    rg = np.random.default_rng(42)
    idx = np.array([rg.choice(len(y), len(y), replace=True) for _ in range(2500)])

    # Turn y and yhat from a pd.Series (vector) to a np.array (matrix) of shape (n, 2500)
    y_ = np.array([y.iloc[i] for i in idx])
    yhat_proba_ = np.array([yhat_proba.iloc[i] for i in idx])

    # Calculate the AUC for each bootstrap sample
    aucs = np.array([roc_auc_score(y_[i], yhat_proba_[i]) for i in range(2500)])

    # Calculate the mean and standard deviation of the AUCs
    logging.info(f"Mean AUC: {np.mean(aucs)}, std AUC: {np.std(aucs)}")
    return np.mean(aucs), np.std(aucs)


@pytest.fixture
def large_sample_data():
    """Generate larger sample data for testing."""
    rg = np.random.default_rng(42)
    size = 100000  # Increase the size for a more realistic test case
    yhat_proba_logits = pd.Series(rg.random(size))
    yhat_proba = 1 / (1 + np.exp(-yhat_proba_logits))
    y = pd.Series(rg.binomial(1, yhat_proba, size=size))
    fold = pd.Series(rg.choice([1, 2, 3, 4, 5], size=size))
    return y, yhat_proba, fold


def test_empirical_auc_variance_basic(
    sample_data: Tuple[pd.Series, pd.Series, pd.Series],
):
    """Test basic functionality of the function."""
    y, yhat_proba, fold = sample_data
    var_auc = _empirical_auc_variance(y, yhat_proba, fold, False)
    assert isinstance(
        var_auc, float
    ), f"Variance should be a float, not {type(var_auc)}"


def test_empirical_auc_variance_correct_calculation(
    sample_data: Tuple[pd.Series, pd.Series, pd.Series], sample_variance: float
):
    """Test the function against a known value."""
    y, yhat_proba, fold = sample_data
    expected_variance = sample_variance
    var_auc = _empirical_auc_variance(y, yhat_proba, fold, False)
    np.testing.assert_almost_equal(var_auc, expected_variance, decimal=2)
    assert (
        np.abs(np.round(var_auc, 2) - np.round(expected_variance, 2)) <= 0.01
    ), f"Expected variance: {expected_variance}, actual variance: {var_auc}"


@pytest.mark.parametrize(
    "y, yhat_proba, fold",
    [
        (pd.Series([]), pd.Series([]), pd.Series([])),  # Empty inputs
        (pd.Series([1, 0]), pd.Series([0.5]), pd.Series([1])),  # Mismatched lengths
    ],
)
def test_invalid_inputs(y: pd.Series, yhat_proba: pd.Series, fold: pd.Series):
    """Test that the function raises an error when inputs are invalid."""
    with pytest.raises(ValueError) as excinfo:
        _empirical_auc_variance(y, yhat_proba, fold, False)

    expected_error = (
        "empirical variance of the AUC estimator cannot be computed if any of the folds have only one class"
        if len(get_unique(y)) > 0
        else "Invalid input"
    )
    expected_error = (
        "empirical variance of the AUC estimator cannot be computed with only one fold"
        if len(get_unique(fold)) == 1
        else expected_error
    )
    expected_error = (
        "nput arrays are empty"
        if (len(y) == 0) | (len(yhat_proba) == 0) | (len(fold) == 0)
        else expected_error
    )

    assert expected_error in str(
        excinfo.value
    ), f"Error message: {excinfo.value}, expected to contain '{expected_error}'"


def test_empirical_auc_variance_with_bootstrapping(
    sample_data: Tuple[pd.Series, pd.Series, pd.Series],
    sample_variance_bootstrap: Tuple[float, float],
):
    """Test variance computation with bootstrapping enabled."""
    y, yhat_proba, fold = sample_data
    m, s = sample_variance_bootstrap
    var_auc_with_bootstrap = _empirical_auc_variance(y, yhat_proba, fold, False)
    # Check that the variance is within 2 standard deviations of the empirical variance
    var_auc = _empirical_auc_variance(y, yhat_proba, fold, False)
    assert (
        (var_auc_with_bootstrap > var_auc - 2 * s)
        & (var_auc_with_bootstrap < var_auc + 2 * s)
    ), f"Variance: {var_auc_with_bootstrap}, empirical variance: {var_auc}, standard deviation: {s}"


def test_imbalanced_classes_in_fold(
    sample_data: Tuple[pd.Series, pd.Series, pd.Series],
):
    """Test that the function handles imbalanced class distributions within folds."""
    y, yhat_proba, fold = sample_data
    # Make an extremely imbalanced distribution
    y.iloc[:95] = 0
    y.iloc[95:] = 1
    fold.iloc[:50] = 1  # Assign a majority of one class to a single fold
    fold.iloc[50:] = 2  # Assign the minority of the other class to a single fold
    fold.iloc[95:97] = 1  # Make sure each class is represented in each fold
    var_auc = _empirical_auc_variance(y, yhat_proba, fold, False)
    assert isinstance(
        var_auc, float
    ), "Expected variance to be a float for imbalanced classes within folds."


def test_large_data_performance(
    large_sample_data: Tuple[pd.Series, pd.Series, pd.Series],
):
    """Test the function's performance on a larger dataset."""
    y, yhat_proba, fold = large_sample_data
    var_auc = _empirical_auc_variance(y, yhat_proba, fold, False)
    assert isinstance(var_auc, float), "Variance should be a float on large datasets."


@pytest.mark.parametrize(
    "y_mod, yhat_proba_mod, fold_mod",
    [
        (
            pd.Series([np.nan, 1, 0]),
            pd.Series([0.5, 0.5, 0.5]),
            pd.Series([1, 2, 2]),
        ),  # NaN in y
        (
            pd.Series([1, 0]),
            pd.Series([np.nan, 0.5]),
            pd.Series([1, 2]),
        ),  # NaN in yhat_proba
        (
            pd.Series([1, 0]),
            pd.Series([0.5, 0.5]),
            pd.Series([np.nan, 2]),
        ),  # NaN in fold
        (
            pd.Series(["a", "b", "C"]),
            pd.Series([0.5, 0.5, 0.5]),
            pd.Series([1, 2, 2]),
        ),  # Non-numeric y
    ],
)
def test_handling_non_numeric_missing_values(
    y_mod: pd.Series, yhat_proba_mod: pd.Series, fold_mod: pd.Series
):
    """Test the function's handling of non-numeric and missing values."""
    with pytest.raises(ValueError):
        _empirical_auc_variance(y_mod, yhat_proba_mod, fold_mod, False)


@pytest.mark.parametrize(
    "skewness",
    [
        (0.99),  # Extreme skew towards positive class
        (0.01),  # Extreme skew towards negative class
    ],
)
def test_skewed_probability_distributions(
    sample_data: Tuple[pd.Series, pd.Series, pd.Series], skewness: float
):
    """Test variance calculation with skewed probability distributions."""
    y, yhat_proba, fold = sample_data
    yhat_proba = yhat_proba.apply(lambda x: min(max(x, skewness), 1 - skewness))
    var_auc = _empirical_auc_variance(y, yhat_proba, fold, False)
    assert isinstance(
        var_auc, float
    ), "Expected variance to be a float for skewed distributions."


@pytest.mark.parametrize("perfect_separation", [(True,), (False,)])
def test_perfect_separation(
    sample_data: Tuple[pd.Series, pd.Series, pd.Series], perfect_separation: bool
):
    """Test variance calculation with perfect or near-perfect separation."""
    y, yhat_proba, fold = sample_data
    if perfect_separation:
        # Simulate perfect separation
        yhat_proba = pd.Series(np.where(y == 1, 0.95, 0.05))
    else:
        # Simulate near-perfect separation
        yhat_proba = pd.Series(np.where(y == 1, 0.9, 0.1))
    var_auc = _empirical_auc_variance(y, yhat_proba, fold, False)
    assert isinstance(
        var_auc, float
    ), f"Expected variance to be a float for (near-)perfect separation, not {type(var_auc)}"


def test_empirical_auc_variance_multi_fold():
    # Generate mock data
    rg = np.random.default_rng(42)
    y = pd.Series(rg.integers(0, 2, 100))
    yhat_proba = pd.Series(rg.random(100))
    fold = pd.Series(rg.choice([1, 2, 3, 4, 5], 100))

    # Call the function
    var_auc = _empirical_auc_variance(y, yhat_proba, fold, False)

    # Assert the variance is a float and non-negative
    assert isinstance(var_auc, float)
    assert var_auc >= 0
