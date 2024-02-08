# import pandas as pd # type: ignore
# import pytest # type: ignore
# from pandas.testing import assert_frame_equal, assert_series_equal # type: ignore

# from predictables.core.src._UnivariateAnalysis import UnivariateAnalysis


# @pytest.fixture
# def df():
#     return pd.read_parquet("cancerdf.parquet")


# @pytest.fixture
# def features(df):
#     return ["comp_1", "comp_2"]


# @pytest.fixture
# def build_ua(df, features):
#     def _make_ua(model_name, has_time_series_structure):
#         return UnivariateAnalysis(
#             model_name=model_name,
#             df_train=df,
#             df_val=df,
#             target_column_name="target",
#             feature_column_names=features,
#             cv_folds=df["cv"],
#             has_time_series_structure=has_time_series_structure,
#         )

#     return _make_ua


# @pytest.mark.parametrize("has_time_series_structure", [True, False])
# def test_univariate_analysis_constructor_model_name(
#     has_time_series_structure, build_ua
# ):
#     model_name = "TestModel"
#     ua = build_ua(model_name, has_time_series_structure)
#     # Test the basic attribute assignment
#     assert ua.model_name == "TestModel", f"model_name={ua.model_name}, not 'TestModel'"


# @pytest.mark.parametrize("has_time_series_structure", [True, False])
# def test_univariate_analysis_constructor_df(build_ua, has_time_series_structure, df):
#     ua = build_ua("TestModel", has_time_series_structure)
#     assert (
#         df.shape[0] == 569
#     ), f"There are 569 rows in the breast cancer dataset, but ua.df.shape[0]={ua.df.shape[0]}, and the fixture df has shape {df.shape}"
#     assert (
#         ua.df.shape[0] == 569
#     ), f"There are 569 rows in the breast cancer dataset, but ua.df.shape[0]={ua.df.shape[0]}, and the fixture df has shape {df.shape}"

#     assert (
#         df.shape[1] == 4
#     ), f"There are 4 columns in my matrix-decomposed breast cancer dataset, but ua.df.shape[1]={ua.df.shape[1]}, and the fixture df has shape {df.shape}"
#     assert (
#         ua.df.shape[1] == 4
#     ), f"There are 4 columns in my matrix-decomposed breast cancer dataset, but ua.df.shape[1]={ua.df.shape[1]}, and the fixture df has shape {df.shape}"

#     assert (
#         "target" in df.columns
#     ), f"target column not in original df:\noriginal: {df}\nnew: {ua.df}"
#     assert (
#         "target" in ua.df.columns
#     ), f"target column not in ua.df:\noriginal: {df}\nnew: {ua.df}"
#     assert (
#         "comp_1" in df.columns
#     ), f"comp_1 column not in original df:\noriginal: {df}\nnew: {ua.df}"
#     assert (
#         "comp_1" in ua.df.columns
#     ), f"comp_1 column not in ua.df:\noriginal: {df}\nnew: {ua.df}"
#     assert (
#         "comp_2" in df.columns
#     ), f"comp_2 column not in original df:\noriginal: {df}\nnew: {ua.df}"
#     assert (
#         "comp_2" in ua.df.columns
#     ), f"comp_2 column not in ua.df:\noriginal: {df}\nnew: {ua.df}"
#     assert (
#         "cv" in df.columns
#     ), f"cv column not in original df:\noriginal: {df}\nnew: {ua.df}"
#     assert (
#         "cv" in ua.df.columns
#     ), f"cv column not in ua.df:\noriginal: {df}\nnew: {ua.df}"

#     (
#         assert_frame_equal(df, ua.df),
#         f"df not equal to original df:\noriginal: {df}\nnew: {ua.df}",
#     )
#     (
#         assert_frame_equal(df, ua.df),
#         f"df not equal to original df:\noriginal: {df}\nnew: {ua.df}",
#     )
#     (
#         assert_frame_equal(df, ua.df_val),
#         f"df_val not equal to original df:\noriginal: {df}\nnew: {ua.df_val}",
#     )


# @pytest.mark.parametrize("has_time_series_structure", [True, False])
# def test_univariate_analysis_constructor_target_column_name(
#     has_time_series_structure, build_ua, features
# ):
#     ua = build_ua("TestModel", has_time_series_structure)
#     assert (
#         ua.target_column_name == "target"
#     ), f"target_column_name={ua.target_column_name}, not 'target'"


# @pytest.mark.parametrize("has_time_series_structure", [True, False])
# def test_univariate_analysis_constructor_feature_column_names(
#     has_time_series_structure, build_ua, features, df
# ):
#     ua = build_ua("TestModel", has_time_series_structure)
#     assert (
#         ua.feature_column_names[0] == "comp_1"
#     ), f"first of feature_column_names={ua.feature_column_names[0]}, not 'comp_1'"
#     assert (
#         ua.feature_column_names[1] == "comp_2"
#     ), f"second of feature_column_names={ua.feature_column_names[1]}, not 'comp_2'"
#     assert (
#         len(ua.feature_column_names) == 2
#     ), f"len(feature_column_names)={len(ua.feature_column_names)}, not 2 (for comp_1 and comp_2)"


# @pytest.mark.parametrize("has_time_series_structure", [True, False])
# def test_univariate_analysis_constructor_cv_folds(
#     has_time_series_structure, build_ua, df
# ):
#     ua = build_ua("TestModel", has_time_series_structure)
#     assert_series_equal(
#         df["cv"],
#         ua.cv_folds,
#         f"cv_folds not equal to original df:\noriginal: {df['cv']}\nnew: {ua.cv_folds}",
#     )
#     assert (
#         sorted(set(ua.cv_folds)) == [1, 2, 3, 4, 5]
#     ), f"the unique values in cv_folds['cv']={ua.cv_folds}, not [1, 2, 3, 4, 5] as expected"


# @pytest.mark.parametrize("has_time_series_structure", [True, False])
# def test_univariate_analysis_constructor_has_time_series_structure(
#     has_time_series_structure, build_ua
# ):
#     ua = build_ua("TestModel", has_time_series_structure)
#     if has_time_series_structure:
#         assert (
#             ua.has_time_series_structure
#         ), f"has_time_series_structure={ua.has_time_series_structure}, not True"
#     else:
#         assert (
#             not ua.has_time_series_structure
#         ), f"has_time_series_structure={ua.has_time_series_structure}, not False"


# @pytest.mark.parametrize("has_time_series_structure", [True, False])
# def test_univariate_analysis_constructor_feature_list(
#     has_time_series_structure, build_ua, features
# ):
#     ua = build_ua("TestModel", has_time_series_structure)
#     assert (
#         len(ua._feature_list) == 2
#     ), f"len(ua._feature_list)={len(ua._feature_list)}, not 2"
#     assert (
#         ua._feature_list[0] == "comp_1"
#     ), f"ua._feature_list[0]={ua._feature_list[0]}, not 'comp_1'"
#     assert (
#         ua._feature_list[1] == "comp_2"
#     ), f"ua._feature_list[1]={ua._feature_list[1]}, not 'comp_2'"
#     assert (
#         ua._feature_list == features
#     ), f"ua._feature_list={ua._feature_list}, not {features}"
