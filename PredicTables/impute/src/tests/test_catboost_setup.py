import pytest
import pandas as pd
from pandas.testing import assert_frame_equal
import polars as pl
import numpy as np

from catboost import CatBoostRegressor, CatBoostClassifier

from PredicTables.impute import initial_impute
from PredicTables.util import to_pd_df, to_pl_df, to_pl_s

# from iterativerf_plan import fit_catboost_models, _prep_data_for_catboost


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


def test_prep_data_for_catboost_with_pddf(pd_df_NO_MISSING):
    # Test with Pandas DataFrame
    expecteddf = pd_df_NO_MISSING
    X_in, y_in = (
        pd_df_NO_MISSING.drop(columns=["num_col1"]),
        pd_df_NO_MISSING["num_col1"],
    )
    X, y = _prep_data_for_catboost(X_in, y_in)
    assert isinstance(X, pd.DataFrame), f"{type(X)} is not a Pandas DataFrame"
    assert isinstance(y, pd.Series), f"{type(y)} is not a Pandas Series"
    assert X.shape == (9, 3), f"X has shape {X.shape}, but should have shape (9, 3)"
    assert y.shape == (9,), f"y has shape {y.shape}, but should have shape (9,)"

    assert_frame_equal(X, expecteddf.drop(columns=["num_col1"]))
    assert_frame_equal(y.to_frame(), expecteddf[["num_col1"]])


def test_prep_data_for_catboost_with_pldf(pd_df_NO_MISSING):
    # Test with Polars DataFrame
    expecteddf = pd_df_NO_MISSING
    X_in, y_in = (
        to_pl_df(pd_df_NO_MISSING.drop(columns=["num_col1"])),
        to_pl_s(pd_df_NO_MISSING["num_col1"]),
    )
    X, y = _prep_data_for_catboost(X_in, y_in)
    assert isinstance(X, pd.DataFrame), f"{type(X)} is not a Pandas DataFrame"
    assert isinstance(y, pd.Series), f"{type(y)} is not a Pandas Series"
    assert X.shape == (9, 3), f"X has shape {X.shape}, but should have shape (9, 3)"
    assert y.shape == (9,), f"y has shape {y.shape}, but should have shape (9,)"

    assert_frame_equal(X, expecteddf.drop(columns=["num_col1"]))
    assert_frame_equal(y.to_frame(), expecteddf[["num_col1"]])


def test_prep_data_for_catboost_with_pddf_missing_num1(pd_df_missing_num1):
    # Test with missing values in num_col1
    expecteddf = to_pd_df(initial_impute(pd_df_missing_num1))
    X, y = _prep_data_for_catboost(
        pd_df_missing_num1.drop(columns=["num_col1"]),
        pd_df_missing_num1["num_col1"],
    )
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert X.shape == (9, 3)
    assert y.shape == (9,)

    assert_frame_equal(X, expecteddf.drop(columns=["num_col1"]))
    assert_frame_equal(y.to_frame(), expecteddf[["num_col1"]])


def test_prep_data_for_catboost_with_pddf_missing_num2(pd_df_missing_num2):
    # Test with missing values in num_col2
    expecteddf = to_pd_df(initial_impute(pd_df_missing_num2))
    X, y = _prep_data_for_catboost(
        pd_df_missing_num2.drop(columns=["num_col2"]),
        pd_df_missing_num2["num_col2"],
    )
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert X.shape == (9, 3)
    assert y.shape == (9,)

    assert_frame_equal(X, expecteddf.drop(columns=["num_col2"]))
    assert_frame_equal(y.to_frame(), expecteddf[["num_col2"]])


def test_prep_data_for_catboost_with_pddf_missing_cat1(pd_df_missing_cat1):
    # Test with missing values in cat_col1
    expecteddf = to_pd_df(initial_impute(pd_df_missing_cat1))
    X, y = _prep_data_for_catboost(
        pd_df_missing_cat1.drop(columns=["cat_col1"]),
        pd_df_missing_cat1["cat_col1"],
    )
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert X.shape == (9, 3)
    assert y.shape == (9,)

    assert_frame_equal(X, expecteddf.drop(columns=["cat_col1"]))
    assert_frame_equal(y.to_frame(), expecteddf[["cat_col1"]])


def test_prep_data_for_catboost_with_pddf_missing_cat2(pd_df_missing_cat2):
    # Test with missing values in cat_col2
    expecteddf = to_pd_df(initial_impute(pd_df_missing_cat2))
    X, y = _prep_data_for_catboost(
        pd_df_missing_cat2.drop(columns=["cat_col2"]),
        pd_df_missing_cat2["cat_col2"],
    )
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert X.shape == (9, 3)
    assert y.shape == (9,)

    assert_frame_equal(X, expecteddf.drop(columns=["cat_col2"]))
    assert_frame_equal(y.to_frame(), expecteddf[["cat_col2"]])


def test_fit_catboost_models_with_numerical_target(pd_df_missing_num1):
    X = pd_df_missing_num1.drop(columns=["num_col1"])
    y = pd_df_missing_num1["num_col1"]

    model = fit_catboost_models(X, y)
    assert isinstance(
        model, CatBoostRegressor
    ), f"{type(model)} is not a CatBoostRegressor, even though y is {y.dtype}"
    # Optionally, add more assertions to test the model's predictions


def test_fit_catboost_models_with_categorical_target(pd_df_missing_cat1):
    X = pd_df_missing_cat1.drop(columns=["cat_col1"])
    y = pd_df_missing_cat1["cat_col1"]

    model = fit_catboost_models(X, y)
    assert isinstance(
        model, CatBoostClassifier
    ), f"{type(model)} is not a CatBoostClassifier, even though y is {y.dtype}"
    assert model.is_fitted(), f"The model ({model}) is not fitted."


def test_fit_catboost_models_with_pandas_dataframe(pd_df_missing_num1):
    X = pd_df_missing_num1.drop(columns=["num_col1"])
    y = pd_df_missing_num1["num_col1"]
    model = fit_catboost_models(X, y)
    assert isinstance(
        model, CatBoostRegressor
    ), f"{type(model)} is not a CatBoostRegressor, even though y is {y.dtype}"
    assert model.is_fitted(), f"The model ({model}) is not fitted."


def test_fit_catboost_models_with_polars_dataframe(pd_df_missing_num1):
    X_pl = to_pl_df(pd_df_missing_num1.drop(columns=["num_col1"]))
    y_pl = to_pl_s(pd_df_missing_num1["num_col1"])
    model = fit_catboost_models(X_pl, y_pl)
    assert isinstance(
        model, CatBoostRegressor
    ), f"{type(model)} is not a CatBoostRegressor, even though y is {y_pl.dtype}"
    assert model.is_fitted(), f"The model ({model}) is not fitted."


def test_fit_catboost_models_with_hyperparameters(pd_df_missing_num1):
    X = pd_df_missing_num1.drop(columns=["num_col1"])
    y = pd_df_missing_num1["num_col1"]
    hyperparameters = {"iterations": 10, "learning_rate": 0.1}
    model = fit_catboost_models(X, y, hyperparameters=hyperparameters)
    assert isinstance(
        model, CatBoostRegressor
    ), f"{type(model)} is not a CatBoostRegressor, even though y is {y.dtype}"
    assert model.is_fitted(), f"The model ({model}) is not fitted."


def test_fit_catboost_models_with_categorical_features(pd_df_missing_num1):
    X = pd_df_missing_num1.drop(columns=["num_col1"])
    y = pd_df_missing_num1["num_col1"]
    cat_features = ["cat_col1", "cat_col2"]
    model = fit_catboost_models(X, y, cat_features=cat_features)
    assert isinstance(
        model, CatBoostRegressor
    ), f"{type(model)} is not a CatBoostRegressor, even though y is {y.dtype}"
    assert model.is_fitted(), f"The model ({model}) is not fitted."


def test_fit_catboost_models_with_categorical_target(pd_df_missing_cat1):
    X = pd_df_missing_cat1.drop(columns=["cat_col1"])
    y_cat = pd_df_missing_cat1["cat_col1"]
    model = fit_catboost_models(X, y_cat)
    assert isinstance(
        model, CatBoostClassifier
    ), f"{type(model)} is not a CatBoostClassifier, even though y is {y_cat.dtype}"
    assert model.is_fitted(), f"The model ({model}) is not fitted."


def test_fit_catboost_models_with_missing_values_in_num_col1(pd_df_missing_num1):
    X = pd_df_missing_num1.drop(columns=["num_col1"])
    y = pd_df_missing_num1["num_col1"]
    model = fit_catboost_models(X, y)
    assert isinstance(
        model, CatBoostRegressor
    ), f"{type(model)} is not a CatBoostRegressor, even though y is {y.dtype}"
    assert model.is_fitted(), f"The model ({model}) is not fitted."


def test_fit_catboost_models_with_missing_values_in_num_col2(pd_df_missing_num2):
    X = pd_df_missing_num2.drop(columns=["num_col2"])
    y = pd_df_missing_num2["num_col2"]
    model = fit_catboost_models(X, y)
    assert isinstance(
        model, CatBoostRegressor
    ), f"{type(model)} is not a CatBoostRegressor, even though y is {y.dtype}"
    assert model.is_fitted(), f"The model ({model}) is not fitted."


def test_fit_catboost_models_with_missing_values_in_cat_col1(pd_df_missing_cat1):
    X = pd_df_missing_cat1.drop(columns=["cat_col1"])
    y = pd_df_missing_cat1["cat_col1"]
    model = fit_catboost_models(X, y)
    assert isinstance(
        model, CatBoostClassifier
    ), f"{type(model)} is not a CatBoostClassifier, even though y is {y.dtype}"
    assert model.is_fitted(), f"The model ({model}) is not fitted."


def test_fit_catboost_models_with_missing_values_in_cat_col2(pd_df_missing_cat2):
    X = pd_df_missing_cat2.drop(columns=["cat_col2"])
    y = pd_df_missing_cat2["cat_col2"]
    model = fit_catboost_models(X, y)
    assert isinstance(
        model, CatBoostClassifier
    ), f"{type(model)} is not a CatBoostClassifier, even though y is {y.dtype}"
    assert model.is_fitted(), f"The model ({model}) is not fitted."
