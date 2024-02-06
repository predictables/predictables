# # FILEPATH: /home/aweaver/work/hit-ratio-model/PredicTables/PCA/src/tests/test_bootstrap_pca.py

# import pytest
# import numpy as np
# import pandas as pd
# from PredicTables.util import to_pl_df


def test_nothing():
    assert 1 == 1


# @pytest.fixture
# def mock_data():
#     # Create mock data
#     cols = ["col_" + str(i) for i in range(10)]
#     data = to_pl_df(pd.DataFrame(np.random.rand(100, 10), columns=cols))
#     n_components = 5
#     return data, n_components

# def test_bootstrap_pca(mock_data):
#     """Test that bootstrap_pca is performed correctly"""
#     # Mock data
#     data, n_components = mock_data

#     # Perform bootstrap_pca
#     bootstrapped_results = bootstrap_pca(data, n_components, n_bootstraps=50)

#     # Assert the number of bootstrapped samples
#     assert (
#         len(bootstrapped_results["loadings"]) == 50
#     ), f"Number of bootstrapped samples ({len(bootstrapped_results['loadings'])}) is incorrect - expected 50"
#     assert (
#         len(bootstrapped_results["explained_variance"]) == 50
#     ), f"Number of bootstrapped samples ({len(bootstrapped_results['explained_variance'])}) is incorrect - expected 50"

#     # Assert the number of components
#     assert (
#         len(bootstrapped_results["loadings"][0]) == n_components
#     ), f"Number of components ({len(bootstrapped_results['loadings'][0])}) is incorrect - expected {n_components}"
#     assert (
#         len(bootstrapped_results["explained_variance"][0]) == n_components
#     ), f"Number of components ({len(bootstrapped_results['explained_variance'][0])}) is incorrect - expected {n_components}"

# def test_invalid_data():
#     """Test that the function raises an error when the data is not a 2-dimensional array"""
#     # Create invalid mock data (1D array)
#     data = pd.DataFrame(np.random.rand(100), columns=["col_0"])
#     n_components = 5

#     with pytest.raises(ValueError):
#         _ = bootstrap_pca(data, n_components)

# def test_invalid_n_components(mock_data):
#     """Test that the function raises an error when the number of components is greater than the number of features"""
#     # Mock data
#     data, _ = mock_data

#     with pytest.raises(ValueError):
#         _ = bootstrap_pca(data, 11)

# def test_bootstrap_pca_stability(mock_data):
#     data, n_components = mock_data
#     bootstrapped_results = bootstrap_pca(data, n_components, n_bootstraps=100)

#     # Calculate variance of loadings across bootstrapped samples
#     loadings_var = np.var(np.array(bootstrapped_results["loadings"]), axis=0)

#     # Assert low variability (indicative of stability)
#     assert np.all(
#         loadings_var < 0.1
#     ), "High variability in PCA loadings across bootstrapped samples indicates instability."
