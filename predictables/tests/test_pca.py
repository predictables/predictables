import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA as sk_pca

from predictables.pca import PCA


@pytest.fixture
def cancer():
    return load_breast_cancer()


@pytest.fixture
def df(cancer):
    return pd.DataFrame(cancer.data, columns=cancer.feature_names)


@pytest.fixture
def skpca(df):
    pca = sk_pca(n_components=2)
    pca.fit(df)
    return pca


@pytest.fixture
def pca(df):
    return PCA(df=df)


def test_pca_initialization(pca):
    """Test that PCA object is initialized correctly"""
    assert pca is not None, f"PCA object: {pca} is None"


def test_pca_n_components(pca):
    """Test that PCA object has n_components attribute as expected"""
    assert (
        pca.n_components is not None
    ), f"PCA object: {pca} has no n_components attribute: {pca.n_components}"
    assert (
        pca.n_components > 0
    ), f"PCA object: {pca} has n_components <= 0: {pca.n_components}"
    assert (
        pca.n_components == 10
    ), f"PCA object: {pca} should default to 10 components, not {pca.n_components}"


def test_pca_df(pca, df):
    """Test that PCA object has df attribute as expected"""
    assert pca.df is not None, f"PCA object: {pca} has no df attribute: {pca.df}"
    assert pca.df.shape[0] > 0, f"PCA object: {pca} has no rows"
    assert pca.df.shape[1] > 0, f"PCA object: {pca} has no columns"
    # assert pca.df is the same df you get from cancer.data
    (
        assert_frame_equal(pca.df, df, check_dtype=False),
        f"PCA object: {pca} has a different df than the one passed in: {df}",
    )


def test_pca_preprocess_data_attr(pca):
    """Test that PCA object has preprocess_data attribute as expected"""
    assert (
        pca.preprocess_data is not None
    ), f"PCA object: {pca} has no preprocess_data attribute: {pca.preprocess_data}"
    assert pca.preprocess_data, f"PCA object: {pca} should default to having preprocess_data == True, not {pca.preprocess_data}"


def test_pca_plotting_backend_attr(pca):
    """Test that PCA object has plotting_backend attribute as expected"""
    assert (
        pca.plotting_backend is not None
    ), f"PCA object: {pca} has no plotting_backend attribute: {pca.plotting_backend}"
    assert (
        pca.plotting_backend == "matplotlib"
    ), f"PCA object: {pca} should default to having plotting_backend == 'matplotlib', not {pca.plotting_backend}"


def test_pca_random_seed_attr(pca):
    """Test that PCA object has random_seed attribute as expected"""
    assert (
        pca.random_seed is not None
    ), f"PCA object: {pca} has no random_seed attribute: {pca.random_seed}"
    assert (
        pca.random_seed == 42
    ), f"PCA object: {pca} should default to having random_seed == 42, not {pca.random_seed}"
