import numpy as np
import pytest

from .._feature_importance import pca_feature_importance
from .._perform_pca import perform_pca


@pytest.fixture
def pca():
    X = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    np.random.seed(42)
    X += np.random.randn(*X.shape)
    pca = perform_pca(X, n_components=2, return_pca_obj=True)
    return pca


def test_with_pre_fitted_pca(pca):
    feature_importances = pca_feature_importance(pca)
    assert np.all(
        feature_importances >= 0
    ), f"Feature importances must be non-negative, and are {feature_importances}"


def test_normalization(pca):
    feature_importances = pca_feature_importance(pca, normalize=True)
    assert np.isclose(
        np.sum(feature_importances), 1
    ), f"Feature importances must sum to 1, and are {feature_importances}, which sum to {np.sum(feature_importances)}"


def test_no_normalization(pca):
    feature_importances = pca_feature_importance(pca, normalize=False)
    assert np.logical_not(
        np.isclose(np.sum(feature_importances), 1)
    ), f"Feature importances are not likely to sum to 1 when not normalized, and are {feature_importances}, which sum to {np.sum(feature_importances)}"


def test_output_shape(pca):
    feature_importances = pca_feature_importance(pca)
    assert feature_importances.shape == (
        3,
    ), f"Feature importances are {feature_importances}. Expected shape (3,), got {feature_importances.shape}"


def test_non_negative_scores(pca):
    feature_importances = pca_feature_importance(pca)
    assert np.all(
        feature_importances >= 0
    ), f"Feature importances must be non-negative, and are {feature_importances}"
