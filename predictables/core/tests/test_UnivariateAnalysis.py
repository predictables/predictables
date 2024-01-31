# import pandas as pd
# import pytest
# from pandas.testing import assert_frame_equal

# from predictables.core.src._UnivariateAnalysis import UnivariateAnalysis
# from predictables.util import to_pd_df


# # Define fixtures for the inputs
# @pytest.fixture
# def df_train():
#     return pd.DataFrame(
#         {"A": [1, 2, 3], "B": [4, 5, 6], "cv": [1, 1, 2], "target": [0, 1, 0]}
#     )


# @pytest.fixture(params=[["A", "B"], ["A"], ["B"]])
# def feature_column_names(request):
#     return request.param


# @pytest.fixture
# def ua(df_train, feature_column_names):
#     return UnivariateAnalysis(df_train, "target", feature_column_names, "cv", False)


# # Test the __init__ method
# def test_UnivariateAnalysis_init(
#     df_train,
#     feature_column_names,
# ):
#     univariate_analysis = UnivariateAnalysis(
#         df_train,
#         "target",
#         feature_column_names,
#         "cv",
#         False,
#     )

#     (
#         assert_frame_equal(univariate_analysis.df, to_pd_df(df_train)),
#         f"Expected: {to_pd_df(df_train)} but got: {univariate_analysis}",
#     )
#     assert (
#         univariate_analysis.target_column_name == "target"
#     ), f"Expected: 'target' but got: {univariate_analysis.target_column_name}"
#     assert (
#         univariate_analysis.feature_column_names == feature_column_names
#     ), f"Expected: {feature_column_names} but got: {univariate_analysis.feature_column_names}"
#     assert (
#         not univariate_analysis.has_time_series_structure
#     ), f"Expected: False but got: {univariate_analysis.has_time_series_structure}"
#     assert univariate_analysis._feature_list == [
#         name.lower()
#         .replace(" ", "_")
#         .replace("-", "_")
#         .replace("/", "_")
#         .replace("(", "")
#         .replace(")", "")
#         for name in feature_column_names
#     ], f"Expected: {feature_column_names} but got: {univariate_analysis._feature_list}"
