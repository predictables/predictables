import numpy as np
import pandas as pd
import pytest
from sklearn.decomposition import PCA

from predictables.util import to_pd_df, to_pl_df, to_pl_lf

from .._perform_pca import perform_pca
from .._preprocessing import preprocess_data_for_pca
from .._select_principal_components import select_n_components_for_variance


@pytest.fixture
def mock_data():
    # Create mock data
    cols = ["col_" + str(i) for i in range(10)]
    X_train = to_pl_df(pd.DataFrame(np.random.rand(100, 10), columns=cols))
    X_val = to_pl_df(pd.DataFrame(np.random.rand(100, 10), columns=cols))
    X_test = to_pl_df(pd.DataFrame(np.random.rand(100, 10), columns=cols))
    n_components = 5
    return X_train, X_val, X_test, n_components


def test_pca_transformation(mock_data):
    """Test that PCA transformation is performed correctly"""
    # Mock data
    X_train, X_val, X_test, n_components = mock_data
    X_train = pd.DataFrame(X_train)
    X_val = pd.DataFrame(X_val)
    X_test = pd.DataFrame(X_test)

    # Perform PCA
    X_train_pca, X_val_pca, X_test_pca = perform_pca(
        to_pd_df(X_train), to_pd_df(X_val), to_pd_df(X_test), n_components
    )

    # Assert the number of components
    assert (
        pd.DataFrame(X_train_pca).shape[1] == n_components
    ), "PCA transformation did not produce the correct number of components"
    assert (
        pd.DataFrame(X_val_pca).shape[1] == n_components
    ), "PCA transformation did not produce the correct number of components"
    assert (
        pd.DataFrame(X_test_pca).shape[1] == n_components
    ), "PCA transformation did not produce the correct number of components"


def test_correct_number_of_components(mock_data):
    """Test that the correct number of components are returned"""
    # Create mock data
    X_train, X_val, X_test, n_components = mock_data

    # Perform PCA
    X_train_pca, X_val_pca, X_test_pca = perform_pca(
        to_pd_df(X_train), to_pd_df(X_val), to_pd_df(X_test), n_components
    )

    # Assert the number of components for each transformed dataset
    assert (
        X_train_pca.shape[1] == n_components
    ), f"PCA on training data did not produce the correct number of components -- Expected {n_components}, got {X_train_pca.shape[1]}"
    assert (
        X_val_pca.shape[1] == n_components
    ), f"PCA on validation data did not produce the correct number of components -- Expected {n_components}, got {X_val_pca.shape[1]}"
    assert (
        X_test_pca.shape[1] == n_components
    ), f"PCA on test data did not produce the correct number of components -- Expected {n_components}, got {X_test_pca.shape[1]}"


def test_data_integrity(mock_data):
    """Test that the original data is not altered after PCA"""
    # Create mock data
    X_train, X_val, X_test, n_components = mock_data

    X_train_original = to_pd_df(X_train).copy()
    X_val_original = to_pd_df(X_val).copy()
    X_test_original = to_pd_df(X_test).copy()

    # Perform PCA
    _ = perform_pca(X_train, X_val, X_test, n_components, return_pca_obj=True)

    # Assert original data is unchanged
    np.testing.assert_array_equal(
        X_train, X_train_original, err_msg="Training data altered after PCA"
    )
    np.testing.assert_array_equal(
        X_val, X_val_original, err_msg="Validation data altered after PCA"
    )
    np.testing.assert_array_equal(
        X_test, X_test_original, err_msg="Test data altered after PCA"
    )


def test_variance_explained(mock_data):
    """Test that the variance explained by the number of principal components needed to explain 90% is also enough to explain 80%"""
    # Create mock data
    X_train, _, _, _ = mock_data

    # Select number of components for 90% variance (to ensure we are > 80%)
    n_components = select_n_components_for_variance(X_train, variance_threshold=0.9)

    # Perform PCA
    X_train = to_pl_df(pd.DataFrame(X_train))
    pca_func = perform_pca(
        to_pl_lf(X_train),
        to_pl_lf(X_train),
        to_pl_lf(X_train),
        n_components,
        return_pca_obj=True,
    )
    pca_sklearn = PCA(n_components=n_components).fit(to_pl_df(X_train).to_numpy())

    # Assert variance explained
    explained_variance_func = sum(pca_func.explained_variance_ratio_)
    explained_variance_sklearn = sum(pca_sklearn.explained_variance_ratio_)
    assert (
        explained_variance_func > 0.8
    ), f"Explained variance ({explained_variance_func:.1%}) should be high (eg > 0.8) for a significant number of components ({n_components})"
    assert (
        explained_variance_sklearn > 0.8
    ), f"Explained variance ({explained_variance_sklearn:.1%}) should be high (eg > 0.8) for a significant number of components ({n_components})"


def test_error_handling():
    """Test that the function raises an error when the number of components is greater than the number of features"""
    # Create invalid mock data (1D array)
    np.random.seed(42)
    X_train = pd.DataFrame(np.random.rand(100), columns=["col_0"])

    np.random.seed(42)
    X_val = pd.DataFrame(np.random.rand(100), columns=["col_0"])

    np.random.seed(42)
    X_test = pd.DataFrame(np.random.rand(100), columns=["col_0"])

    with pytest.raises(ValueError):
        _ = perform_pca(X_train, X_val, X_test, 5)


def test_pca_comparison_with_sklearn(mock_data):
    X_train, _, _, n_components = mock_data
    X_train_pd = to_pd_df(X_train)

    # Perform PCA using perform_pca
    pca_custom = perform_pca(
        X_train_pd,
        n_components=n_components,
        return_pca_obj=True,
        random_state=42,
        preprocess_data=False,
    )

    # Perform PCA using sklearn
    pca_sklearn = PCA(n_components=n_components, random_state=42)
    pca_sklearn.fit(X_train_pd)

    # Align signs and compare PCA components
    for i in range(n_components):
        # Flip signs if the direction is opposite
        if np.sign(pca_custom.components_[i, 0]) != np.sign(
            pca_sklearn.components_[i, 0]
        ):
            pca_sklearn.components_[i, :] *= -1

    # Compare PCA component magnitudes
    for i, comp in enumerate(pca_custom.components_):
        assert np.sqrt(np.sum(np.power(comp, 2))) == pytest.approx(
            np.sqrt(np.sum(np.power(pca_sklearn.components_[i, :], 2))),
            rel=1e-2,
        ), f"PCA component magnitude is not 1: {np.sqrt(np.sum(np.power(comp, 2)))}"

    # Compare explained variance
    assert pca_custom.explained_variance_ == pytest.approx(
        pca_sklearn.explained_variance_, rel=1e-1
    ), f"Explained variance differs: {pca_custom.explained_variance_} vs {pca_sklearn.explained_variance_}"


def test_pca_reproducibility(mock_data):
    X_train, _, _, n_components = mock_data
    X_train_pd = to_pd_df(X_train)

    # Perform PCA twice with the same random state
    pca1 = perform_pca(
        X_train_pd,
        n_components=n_components,
        random_state=42,
        return_pca_obj=True,
    )
    pca2 = perform_pca(
        X_train_pd,
        n_components=n_components,
        random_state=42,
        return_pca_obj=True,
    )

    # Compare results
    np.testing.assert_array_equal(
        pca1.components_,
        pca2.components_,
        err_msg=f"PCA results are not reproducible with the same random state - components differ:\n{pca1.components_}\nvs\n{pca2.components_}",
    )
    np.testing.assert_array_equal(
        pca1.explained_variance_,
        pca2.explained_variance_,
        err_msg=f"PCA results are not reproducible with the same random state - explained variance differs:\n{pca1.explained_variance_}\nvs\n{pca2.explained_variance_}",
    )
    np.testing.assert_array_equal(
        pca1.explained_variance_ratio_,
        pca2.explained_variance_ratio_,
        err_msg=f"PCA results are not reproducible with the same random state - explained variance ratio differs:\n{pca1.explained_variance_ratio_}\nvs\n{pca2.explained_variance_ratio_}",
    )
    np.testing.assert_array_equal(
        pca1.singular_values_,
        pca2.singular_values_,
        err_msg=f"PCA results are not reproducible with the same random state - singular values differs:\n{pca1.singular_values_}\nvs\n{pca2.singular_values_}",
    )
    np.testing.assert_array_equal(
        pca1.mean_,
        pca2.mean_,
        err_msg=f"PCA results are not reproducible with the same random state - mean differs:\n{pca1.mean_}\nvs\n{pca2.mean_}",
    )
    np.testing.assert_array_equal(
        pca1.noise_variance_,
        pca2.noise_variance_,
        err_msg=f"PCA results are not reproducible with the same random state - noise variance differs:\n{pca1.noise_variance_}\nvs\n{pca2.noise_variance_}",
    )


@pytest.mark.parametrize("data_format", [to_pd_df, to_pl_df])
def test_pca_data_formats(mock_data, data_format):
    X_train, _, _, n_components = mock_data
    X_train_formatted = data_format(X_train)

    # Perform PCA
    pca_result = perform_pca(
        X_train_formatted, n_components=n_components, return_pca_obj=True
    )

    # Validate the result
    assert pca_result.components_.shape == (
        n_components,
        X_train_formatted.shape[1],
    ), f"PCA result shape is incorrect for data format. Expected {(n_components, X_train_formatted.shape[1])}, got {pca_result.components_.shape}"
    assert pca_result.explained_variance_.shape == (
        n_components,
    ), f"PCA result shape is incorrect for data format. Expected {(n_components,)}, got {pca_result.explained_variance_.shape}"
    assert pca_result.explained_variance_ratio_.shape == (
        n_components,
    ), f"PCA result shape is incorrect for data format. Expected {(n_components,)}, got {pca_result.explained_variance_ratio_.shape}"
    assert pca_result.singular_values_.shape == (
        n_components,
    ), f"PCA result shape is incorrect for data format. Expected {(n_components,)}, got {pca_result.singular_values_.shape}"
    assert pca_result.mean_.shape == (
        X_train_formatted.shape[1],
    ), f"PCA result shape is incorrect for data format. Expected {(X_train_formatted.shape[1],)}, got {pca_result.mean_.shape}"
    assert (
        pca_result.noise_variance_.shape == ()
    ), f"PCA result shape is incorrect for data format. Expected {()}, got {pca_result.noise_variance_.shape}"

    # Assert that the result is the same regardless of data format
    pca_result1 = perform_pca(
        X_train_formatted, n_components=n_components, return_pca_obj=True
    )
    pca_result2 = perform_pca(
        data_format(X_train), n_components=n_components, return_pca_obj=True
    )
    np.testing.assert_array_equal(
        pca_result1.components_,
        pca_result2.components_,
        err_msg=f"PCA results differ for different data formats:\n{pca_result1.components_}\nvs\n{pca_result2.components_}",
    )


def test_pca_with_preprocessing(mock_data):
    X_train, _, _, n_components = mock_data
    X_train_processed = preprocess_data_for_pca(X_train)

    # Perform PCA on preprocessed data
    pca_processed = perform_pca(
        X_train_processed, n_components=n_components, return_pca_obj=True
    )

    # Perform PCA on raw data
    pca_raw = perform_pca(X_train, n_components=n_components, return_pca_obj=True)

    # Compare explained variance
    assert not np.array_equal(
        pca_processed.explained_variance_ratio_,
        pca_raw.explained_variance_ratio_,
    ), f"PCA explained variance ratios should differ between preprocessed and raw data:\n{pca_processed.explained_variance_ratio_}\nvs\n{pca_raw.explained_variance_ratio_}"


def test_pca_high_dimensional_data():
    # High-dimensional mock data
    X_train = np.random.rand(100, 100)  # 100 samples, 100 features
    n_components = 10

    # Perform PCA
    pca_result = perform_pca(X_train, n_components=n_components, return_pca_obj=True)

    # Validate the result
    assert pca_result.components_.shape == (
        n_components,
        100,
    ), f"PCA result shape ({pca_result.components_.shape}) is incorrect for high-dimensional data - expected {(n_components, 100)}"
