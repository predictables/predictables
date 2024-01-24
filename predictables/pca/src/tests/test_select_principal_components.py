import numpy as np
from sklearn.datasets import make_classification

from predictables.pca.src._select_principal_components import (
    select_n_components_for_variance,
)


def test_select_n_components_for_variance():
    # Generate a synthetic dataset
    n_features = 20
    X, _ = make_classification(n_samples=100, n_features=n_features, random_state=42)

    # Test for 95% variance threshold
    var_threshold = 0.95
    n_components = select_n_components_for_variance(X, variance_threshold=var_threshold)

    # Assert conditions
    assert isinstance(
        n_components, np.int64
    ), f"The function should return an integer, not {type(n_components)} -- {n_components}"
    assert (
        n_components <= n_features
    ), f"The number of components ({n_components}) should not exceed the number of features ({n_features})"
    assert (
        n_components >= 1
    ), f"The number of components ({n_components}) should be at least 1"
