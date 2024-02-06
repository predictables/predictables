# import numpy as np # type: ignore
# import pandas as pd # type: ignore
# import pytest # type: ignore
# from sklearn.datasets import load_breast_cancer  # type: ignore
# from sklearn.decomposition import PCA  # type: ignore

# # from predictables.pca.src._create_loading_plot_ORIGINAL import (  # type: ignore
# # calculate_cumulative_loading_threshold,
# # prepare_loadings_data,
# # validate_inputs,
# # )


# @pytest.fixture
# def df():
#     cancer = load_breast_cancer()
#     return pd.DataFrame(cancer.data, columns=cancer.feature_names)


# @pytest.fixture
# def pca(df):
#     pca = PCA(n_components=2)
#     pca.fit(df)
#     return pca


# @pytest.fixture
# def feature_names(df):
#     return df.columns


# @pytest.mark.parametrize(
#     "n_components",
#     [
#         2,
#         3,
#         4,
#     ],
# )
# @pytest.mark.parametrize(
#     "max_features",
#     [
#         None,
#         10,
#         15,
#         1,
#         2,
#         3,
#     ],
# )
# @pytest.mark.parametrize(
#     "drop_legend_when_n_features",
#     [
#         10,
#         15,
#         20,
#     ],
# )
# def test_validate_inputs(
#     n_components,
#     max_features,
#     drop_legend_when_n_features,
#     pca,
#     feature_names,
#     expected,
# ):
#     result = validate_inputs(
#         pca, feature_names, n_components, max_features, drop_legend_when_n_features
#     )
#     assert result == expected, f"Expected {expected}, but got {result}"


# @pytest.mark.parametrize(
#     "n_components, average_loading_threshold, expected",
#     [
#         (
#             2,
#             0.05,
#             0.1,
#         ),  # Test case where n_components is 2 and average_loading_threshold is 0.05
#         (
#             3,
#             0.1,
#             0.3,
#         ),  # Test case where n_components is 3 and average_loading_threshold is 0.1
#         (
#             5,
#             0.2,
#             1.0,
#         ),  # Test case where n_components is 5 and average_loading_threshold is 0.2
#     ],
# )
# def test_calculate_cumulative_loading_threshold(
#     n_components, average_loading_threshold, expected
# ):
#     result = calculate_cumulative_loading_threshold(
#         n_components, average_loading_threshold
#     )
#     result = np.round(result, 3)
#     expected = np.round(expected, 3)
#     assert result == expected, f"Expected {expected}, but got {result}"


# @pytest.mark.parametrize(
#     "n_components, expected_shape, expected_columns",
#     [
#         (2, (5, 2), ["PC-01", "PC-02"]),  # Test case where n_components is 2
#         (3, (5, 3), ["PC-01", "PC-02", "PC-03"]),  # Test case where n_components is 3
#     ],
# )
# def test_prepare_loadings_data(
#     pca, feature_names, n_components, expected_shape, expected_columns
# ):
#     df = prepare_loadings_data(pca, feature_names, n_components)

#     assert (
#         df.shape == expected_shape
#     ), f"Expected shape {expected_shape}, but got {df.shape}"
#     assert (
#         list(df.columns) == expected_columns
#     ), f"Expected columns {expected_columns}, but got {list(df.columns)}"
#     assert list(df.index) == list(
#         feature_names
#     ), f"Expected index {list(feature_names)}, but got {list(df.index)}"
