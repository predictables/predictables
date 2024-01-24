import numpy as np
import pandas as pd
import pytest

from predictables.impute.src._train_catboost_model import train_catboost_model


@pytest.fixture
def pd_df_NO_MISSING():
    """
    Sample dataframe for testing.
    num_col1 is a `float`
    num_col2 is an `int`
    cat_col1 is a `category`
    cat_col2 is a `string`
    """
    np.random.seed(42)
    df = pd.DataFrame(
        {
            "num_col1": [1, 2, 3, 4, 5, 6, 7, 8, 9],
            "num_col2": pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9])
            * (1 + np.random.randn(9)),
            "cat_col1": ["a"] * 4 + ["b"] * 3 + ["c"] * 2,
            "cat_col2": ["a"] * 2 + ["b"] * 4 + ["c"] * 3,
        },
        index=range(0, 9),
    )
    df["num_col1"] = df["num_col1"].astype("float")
    df["cat_col1"] = df["cat_col1"].astype("category")

    return df


@pytest.fixture
def pd_df_missing_num1(pd_df_NO_MISSING):
    missing_idx = [1, 3, 5, 7]
    pd_df_missing = pd_df_NO_MISSING
    pd_df_missing.loc[missing_idx, "num_col1"] = np.nan
    return pd_df_missing


@pytest.fixture
def pd_df_missing_num2(pd_df_NO_MISSING):
    missing_idx = [2, 8]
    pd_df_missing = pd_df_NO_MISSING
    pd_df_missing.loc[missing_idx, "num_col2"] = np.nan
    return pd_df_missing


@pytest.fixture
def pd_df_missing_cat1(pd_df_NO_MISSING):
    missing_idx = [1, 5]
    pd_df_missing = pd_df_NO_MISSING
    pd_df_missing.loc[missing_idx, "cat_col1"] = np.nan
    return pd_df_missing


@pytest.fixture
def pd_df_missing_cat2(pd_df_NO_MISSING):
    missing_idx = [3, 4]
    pd_df_missing = pd_df_NO_MISSING
    pd_df_missing.loc[missing_idx, "cat_col2"] = np.nan
    return pd_df_missing


@pytest.fixture
def fitted_model_num1(pd_df_missing_num1):
    """
    Fit a CatBoostRegressor model to predict num_col1.
    """
    df = pd_df_missing_num1
    X = df.drop(columns=["num_col1"])
    y = df["num_col1"]
    catcols = ["cat_col1", "cat_col2"]
    model = train_catboost_model(X, y, catcols)[0]
    return model


@pytest.fixture
def fitted_model_num2(pd_df_missing_num2):
    """
    Fit a CatBoostRegressor model to predict num_col2.
    """
    df = pd_df_missing_num2
    X = df.drop(columns=["num_col2"])
    y = df["num_col2"]
    catcols = ["cat_col1", "cat_col2"]
    model = train_catboost_model(X, y, catcols)[0]
    return model


@pytest.fixture
def fitted_model_cat1(pd_df_missing_cat1):
    """
    Fit a CatBoostClassifier model to predict cat_col1.
    """
    df = pd_df_missing_cat1
    X = df.drop(columns=["cat_col1"])
    y = df["cat_col1"]
    catcols = ["cat_col2"]
    model = train_catboost_model(X, y, catcols)[1]
    return model


@pytest.fixture
def fitted_model_cat2(pd_df_missing_cat2):
    """
    Fit a CatBoostClassifier model to predict cat_col2.
    """
    df = pd_df_missing_cat2
    X = df.drop(columns=["cat_col2"])
    y = df["cat_col2"]
    catcols = ["cat_col1"]
    model = train_catboost_model(X, y, catcols)[1]
    return model


# def test_predict_missing_values_num_col1_pddf(fitted_model_num1, pd_df_missing_num1):
#     # Test with Pandas DataFrame
#     X_features = pd_df_missing_num1.drop(columns=["num_col1"])
#     missing_mask = get_missing_data_mask(pd_df_missing_num1[["num_col1"]])
#     predicted_values = predict_missing_values(
#         fitted_model_num1, X_features, missing_mask
#     )
#     assert isinstance(
#         predicted_values, pl.Series
#     ), f"type(predicted_values): {type(predicted_values)}\nExpected type: pl.Series"
#     assert predicted_values.shape == (
#         4,
#     ), f"predicted_values.shape: {predicted_values.shape}\nExpected shape: (4,)"
#     assert (
#         predicted_values.dtype == "float64"
#     ), f"predicted_values.dtype: {predicted_values.dtype}\nExpected dtype: float64"


# def test_predict_missing_values_num_col1_pldf(fitted_model_num1, pd_df_missing_num1):
#     # Test with Polars DataFrame
#     X_features = to_pl_df(pd_df_missing_num1.drop(columns=["num_col1"]))
#     missing_mask = to_pl_s(get_missing_data_mask(pd_df_missing_num1[["num_col1"]]))
#     predicted_values = predict_missing_values(
#         fitted_model_num1, X_features, missing_mask
#     )
#     assert isinstance(
#         predicted_values, pl.Series
#     ), f"type(predicted_values): {type(predicted_values)}\nExpected type: pl.Series"
#     assert predicted_values.shape == (
#         4,
#     ), f"predicted_values.shape: {predicted_values.shape}\nExpected shape: (4,)"
#     assert (
#         predicted_values.dtype == "float64"
#     ), f"predicted_values.dtype: {predicted_values.dtype}\nExpected dtype: float64"


# def test_predict_missing_values_num_col2_pddf(fitted_model_num2, pd_df_missing_num2):
#     # Test with Pandas DataFrame
#     X_features = pd_df_missing_num2.drop(columns=["num_col2"])
#     missing_mask = get_missing_data_mask(pd_df_missing_num2[["num_col2"]])
#     predicted_values = predict_missing_values(
#         fitted_model_num2, X_features, missing_mask
#     )
#     assert isinstance(
#         predicted_values, pl.Series
#     ), f"type(predicted_values): {type(predicted_values)}\nExpected type: pl.Series"
#     assert predicted_values.shape == (
#         2,
#     ), f"predicted_values.shape: {predicted_values.shape}\nExpected shape: (2,)"
#     assert (
#         predicted_values.dtype == "float64"
#     ), f"predicted_values.dtype: {predicted_values.dtype}\nExpected dtype: float64"


# def test_predict_missing_values_num_col2_pldf(fitted_model_num2, pd_df_missing_num2):
#     # Test with Polars DataFrame
#     X_features = to_pl_df(pd_df_missing_num2.drop(columns=["num_col2"]))
#     missing_mask = to_pl_s(get_missing_data_mask(pd_df_missing_num2[["num_col2"]]))
#     predicted_values = predict_missing_values(
#         fitted_model_num2, X_features, missing_mask
#     )
#     assert isinstance(
#         predicted_values, pl.Series
#     ), f"type(predicted_values): {type(predicted_values)}\nExpected type: pl.Series"
#     assert predicted_values.shape == (
#         2,
#     ), f"predicted_values.shape: {predicted_values.shape}\nExpected shape: (2,)"
#     assert (
#         predicted_values.dtype == "float64"
#     ), f"predicted_values.dtype: {predicted_values.dtype}\nExpected dtype: float64"


# def test_predict_missing_values_cat_col1_pddf(fitted_model_cat1, pd_df_missing_cat1):
#     # Test with Pandas DataFrame
#     X_features = pd_df_missing_cat1.drop(columns=["cat_col1"])
#     missing_mask = get_missing_data_mask(pd_df_missing_cat1[["cat_col1"]])
#     predicted_values = predict_missing_values(
#         fitted_model_cat1, X_features, missing_mask
#     )
#     assert isinstance(
#         predicted_values, pl.Series
#     ), f"type(predicted_values): {type(predicted_values)}\nExpected type: pl.Series"
#     assert predicted_values.shape == (
#         2,
#     ), f"predicted_values.shape: {predicted_values.shape}\nExpected shape: (2,)"
#     assert (
#         predicted_values.dtype == "object"
#     ), f"predicted_values.dtype: {predicted_values.dtype}\nExpected dtype: object"


# def test_predict_missing_values_cat_col1_pldf(fitted_model_cat1, pd_df_missing_cat1):
#     # Test with Polars DataFrame
#     X_features = to_pl_df(pd_df_missing_cat1.drop(columns=["cat_col1"]))
#     missing_mask = to_pl_s(get_missing_data_mask(pd_df_missing_cat1[["cat_col1"]]))
#     predicted_values = predict_missing_values(
#         fitted_model_cat1, X_features, missing_mask
#     )
#     assert isinstance(
#         predicted_values, pl.Series
#     ), f"type(predicted_values): {type(predicted_values)}\nExpected type: pl.Series"
#     assert predicted_values.shape == (
#         2,
#     ), f"predicted_values.shape: {predicted_values.shape}\nExpected shape: (2,)"
#     assert (
#         predicted_values.dtype == "object"
#     ), f"predicted_values.dtype: {predicted_values.dtype}\nExpected dtype: object"


# def test_predict_missing_values_cat_col2_pddf(fitted_model_cat2, pd_df_missing_cat2):
#     # Test with Pandas DataFrame
#     X_features = pd_df_missing_cat2.drop(columns=["cat_col2"])
#     missing_mask = get_missing_data_mask(pd_df_missing_cat2[["cat_col2"]])
#     predicted_values = predict_missing_values(
#         fitted_model_cat2, X_features, missing_mask
#     )
#     assert isinstance(
#         predicted_values, pl.Series
#     ), f"type(predicted_values): {type(predicted_values)}\nExpected type: pl.Series"
#     assert predicted_values.shape == (
#         2,
#     ), f"predicted_values.shape: {predicted_values.shape}\nExpected shape: (2,)"
#     assert (
#         predicted_values.dtype == "object"
#     ), f"predicted_values.dtype: {predicted_values.dtype}\nExpected dtype: object"


# def test_predict_missing_values_cat_col2_pldf(fitted_model_cat2, pd_df_missing_cat2):
#     # Test with Polars DataFrame
#     X_features = to_pl_df(pd_df_missing_cat2.drop(columns=["cat_col2"]))
#     missing_mask = to_pl_s(get_missing_data_mask(pd_df_missing_cat2[["cat_col2"]]))
#     predicted_values = predict_missing_values(
#         fitted_model_cat2, X_features, missing_mask
#     )
#     assert isinstance(
#         predicted_values, pl.Series
#     ), f"type(predicted_values): {type(predicted_values)}\nExpected type: pl.Series"
#     assert predicted_values.shape == (
#         2,
#     ), f"predicted_values.shape: {predicted_values.shape}\nExpected shape: (2,)"
#     assert (
#         predicted_values.dtype == "object"
#     ), f"predicted_values.dtype: {predicted_values.dtype}\nExpected dtype: object"
