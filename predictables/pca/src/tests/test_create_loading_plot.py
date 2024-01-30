# FILEPATH: /home/aweaver/work/predictables/predictables/pca/src/tests/test__create_loading_plot_REFACTOR.py
import numpy as np
import pandas as pd
import pytest
from sklearn.decomposition import PCA  # type: ignore

from predictables.pca.src._create_loading_plot import (
    calculate_cumulative_loading_threshold,
    prepare_loadings_data,
    validate_inputs,
)


@pytest.fixture
def pca():
    return PCA(n_components=3)


@pytest.fixture
def feature_names():
    return np.array(["a", "b", "c", "d", "e"])


@pytest.fixture
def pca2():
    pca = PCA(n_components=3)
    pca.fit(np.random.rand(5, 5))  # Fit the PCA on some random data
    return pca


@pytest.mark.parametrize(
    "n_components, max_features, drop_legend_when_n_features, expected",
    [
        # Test case where n_components is less than pca.n_components and max_features is None
        (2, None, 2, (2, 5, True)),
        # Test case where n_components is more than pca.n_components and max_features is less than feature_names.shape[0]
        (4, 3, 2, (3, 3, True)),
        # Test case where n_components is less than pca.n_components and max_features is less than drop_legend_when_n_features
        (2, 3, 4, (2, 3, False)),
    ],
)
def test_validate_inputs(
    pca,
    feature_names,
    n_components,
    max_features,
    drop_legend_when_n_features,
    expected,
):
    result = validate_inputs(
        pca, feature_names, n_components, max_features, drop_legend_when_n_features
    )
    assert result == expected, f"Expected {expected}, but got {result}"


@pytest.mark.parametrize(
    "n_components, average_loading_threshold, expected",
    [
        (
            2,
            0.05,
            0.1,
        ),  # Test case where n_components is 2 and average_loading_threshold is 0.05
        (
            3,
            0.1,
            0.3,
        ),  # Test case where n_components is 3 and average_loading_threshold is 0.1
        (
            5,
            0.2,
            1.0,
        ),  # Test case where n_components is 5 and average_loading_threshold is 0.2
    ],
)
def test_calculate_cumulative_loading_threshold(
    n_components, average_loading_threshold, expected
):
    result = calculate_cumulative_loading_threshold(
        n_components, average_loading_threshold
    )
    assert result == expected, f"Expected {expected}, but got {result}"


@pytest.mark.parametrize(
    "n_components, expected_shape, expected_columns",
    [
        (2, (5, 2), ["PC-01", "PC-02"]),  # Test case where n_components is 2
        (3, (5, 3), ["PC-01", "PC-02", "PC-03"]),  # Test case where n_components is 3
    ],
)
def test_prepare_loadings_data(
    pca2, feature_names, n_components, expected_shape, expected_columns
):
    df = prepare_loadings_data(pca2, feature_names, n_components)

    assert (
        df.shape == expected_shape
    ), f"Expected shape {expected_shape}, but got {df.shape}"
    assert (
        list(df.columns) == expected_columns
    ), f"Expected columns {expected_columns}, but got {list(df.columns)}"
    assert list(df.index) == list(
        feature_names
    ), f"Expected index {list(feature_names)}, but got {list(df.index)}"
