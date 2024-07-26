import pytest
import numpy as np
import polars as pl
import polars.testing as pltest
import pandas as pd
from predictables.core.src.univariate_config import UnivariateConfig


# Sample data for testing
@pytest.fixture
def sample_data():
    return pl.DataFrame(
        {
            "index": range(100),
            "feature1": range(100),
            "feature2": range(100, 200),
            "target": [1 if i % 2 == 0 else 0 for i in range(100)],
            "cv": [i % 5 for i in range(100)],
        }
    ).lazy()


@pytest.fixture
def univariate_config(sample_data):
    return UnivariateConfig(
        model_name="test_model",
        df_train=sample_data,
        df_val_=sample_data,
        target_column_name="target",
        feature_column_names=["feature1", "feature2"],
        time_series_validation=True,
        cv_folds=pl.Series("cv_folds", [0, 1, 2, 3, 4]),
    )


def test_df_getter_setter(univariate_config, sample_data):
    """Test getter and setter for df property."""
    # assert univariate_config.df is sample_data, f"Expected that "
    pltest.assert_frame_equal(univariate_config.df, sample_data)

    new_df = sample_data.with_columns(pl.col("feature1") * 2)
    univariate_config.df = new_df
    # assert univariate_config.df is new_df
    pltest.assert_frame_equal(univariate_config.df, new_df)


def test_df_val_getter_setter(univariate_config, sample_data):
    """Test getter and setter for df_val property."""
    # assert univariate_config.df_val is sample_data
    pltest.assert_frame_equal(univariate_config.df_val, sample_data)

    new_df = sample_data.with_columns(pl.col("feature2") * 2)
    univariate_config.df_val = new_df
    # assert univariate_config.df_val is new_df
    pltest.assert_frame_equal(univariate_config.df_val, new_df)


def test_features(univariate_config):
    """Test that features property returns correct feature names."""
    assert (
        univariate_config.features == ["feature1", "feature2"]
    ), f"Expected features to be ['feature1', 'feature2'], but got {univariate_config.features}"


@pytest.mark.parametrize("time_series_validation", [True, False])
def test_X_y_by_cv(time_series_validation, sample_data):
    """Test the generator function for training and validation splits."""
    config = UnivariateConfig(
        model_name="test_model",
        df_train=sample_data,
        df_val_=sample_data,
        target_column_name="target",
        feature_column_names=["feature1", "feature2"],
        time_series_validation=time_series_validation,
        cv_folds=pl.Series("cv_folds", [0, 1, 2, 3, 4]),
    )

    for X_train, X_test, y_train, y_test in config.X_y_by_cv:
        assert isinstance(
            X_train, pd.DataFrame
        ), f"Expected X_train to be a pandas DataFrame, but got {type(X_train)}"
        assert isinstance(
            X_test, pd.DataFrame
        ), f"Expected X_test to be a pandas DataFrame, but got {type(X_test)}"
        assert isinstance(
            y_train, np.ndarray
        ), f"Expected y_train to be a numpy array, but got {type(y_train)}"
        assert isinstance(
            y_test, np.ndarray
        ), f"Expected y_test to be a numpy array, but got {type(y_test)}"
        assert (
            "feature1" in X_train.columns
        ), f"Expected 'feature1' in X_train columns, but got {X_train.columns}"
        assert (
            "feature2" in X_train.columns
        ), f"Expected 'feature2' in X_train columns, but got {X_train.columns}"
        assert (
            "feature1" in X_test.columns
        ), f"Expected 'feature1' in X_test columns, but got {X_test.columns}"
        assert (
            "feature2" in X_test.columns
        ), f"Expected 'feature2' in X_test columns, but got {X_test.columns}"
        assert (
            y_train.shape[0] > 0
        ), f"Expected y_train to have more than 0 elements, but got {y_train.shape[0]}"
        assert (
            y_test.shape[0] > 0
        ), f"Expected y_test to have more than 0 elements, but got {y_test.shape[0]}"
        assert (
            # training indices
            set(X_train["index"].tolist())
            # testing indices
            & set(X_test["index"].tolist())
            # intersection of training and testing indices should be empty
            == set()
        ), f"Expected no overlap between X_train and X_test indices, but found overlap: {set(X_train['index'].tolist()) & set(X_test['index'].tolist())}"


def test_filter_for_time_series(univariate_config, sample_data):
    """Test time series filter function."""
    fold = 2
    df_train, df_val = univariate_config.filter_for_time_series(sample_data, fold)
    df_train = df_train.collect().to_pandas()
    df_val = df_val.collect().to_pandas()

    assert all(
        df_train["cv"] < fold
    ), f"Expected all cv values in df_train to be less than {fold} for time series validation, but got {df_train['cv'].unique()}"
    assert all(
        df_val["cv"] == fold
    ), f"Expected all cv values in df_val to be {fold} for time series validation, but got {df_val['cv'].unique()}"
    assert (
        len(df_train) + len(df_val) == 60
    ), f"We use fold {fold} out of (0, 1, 2, 3, 4), so (0, 1, 2) should be 60% of the data. Expected the sum of the lengths of df_train and df_val to be 60, but got {len(df_train) + len(df_val)}"


def test_filter_for_non_time_series(univariate_config, sample_data):
    """Test non-time series filter function."""
    fold = 2
    df_train, df_val = univariate_config.filter_for_non_time_series(sample_data, fold)
    df_train = df_train.collect().to_pandas()
    df_val = df_val.collect().to_pandas()

    assert all(
        df_train["cv"] != fold
    ), f"Expected all cv values in df_train to not be {fold} for non-time series validation, but got {df_train['cv'].unique()}"
    assert all(
        df_val["cv"] == fold
    ), f"Expected all cv values in df_val to be {fold} for non-time series validation, but got {df_val['cv'].unique()}"
    assert (
        len(df_train) + len(df_val) == 100
    ), f"Expected the sum of the lengths of df_train and df_val to be 100, but got {len(df_train) + len(df_val)}"


def test_empty_dataframe():
    """Test with empty DataFrame."""
    empty_df = pl.DataFrame(
        {"feature1": [], "feature2": [], "target": [], "cv": []}
    ).lazy()

    with pytest.raises(ValueError) as e:
        UnivariateConfig(
            model_name="empty_test",
            df_train=empty_df,
            df_val_=empty_df,
            target_column_name="target",
            feature_column_names=["feature1", "feature2"],
            time_series_validation=False,
            cv_folds=pl.Series("cv_folds", [0, 1, 2, 3, 4]),
        )

    assert (
        "Empty dataframes are not supported" in str(e.value)
    ), f"Expected 'Empty dataframes are not supported' in the error message, but got {e.value}"


def test_missing_values():
    """Test with DataFrame containing missing values."""
    data_with_nan = {
        "feature1": [1, 2, None, 4, 5],
        "feature2": [None, 2, 3, 4, 5],
        "target": [1, 0, 1, 0, None],
        "cv": [0, 1, 2, 3, 4],
    }
    df_with_nan = pl.DataFrame(data_with_nan).lazy()
    config = UnivariateConfig(
        model_name="nan_test",
        df_train=df_with_nan,
        df_val_=df_with_nan,
        target_column_name="target",
        feature_column_names=["feature1", "feature2"],
        time_series_validation=False,
        cv_folds=pl.Series("cv_folds", [0, 1, 2, 3, 4]),
    )

    for X_train, X_test, y_train, y_test in config.X_y_by_cv:
        assert isinstance(
            X_train, pd.DataFrame
        ), f"Expected X_train to be a pandas DataFrame, but got {type(X_train)}"
        assert isinstance(
            X_test, pd.DataFrame
        ), f"Expected X_test to be a pandas DataFrame, but got {type(X_test)}"
        assert isinstance(
            y_train, np.ndarray
        ), f"Expected y_train to be a numpy array, but got {type(y_train)}"
        assert isinstance(
            y_test, np.ndarray
        ), f"Expected y_test to be a numpy array, but got {type(y_test)}"


def test_single_fold():
    """Test with a single fold for cross-validation."""
    data_single_fold = {
        "feature1": range(10),
        "feature2": range(10, 20),
        "target": [1 if i % 2 == 0 else 0 for i in range(10)],
        "cv": [0 for _ in range(10)],
    }
    df_single_fold = pl.DataFrame(data_single_fold).lazy()
    config = UnivariateConfig(
        model_name="single_fold_test",
        df_train=df_single_fold,
        df_val_=df_single_fold,
        target_column_name="target",
        feature_column_names=["feature1", "feature2"],
        time_series_validation=False,
        cv_folds=pl.Series("cv_folds", [0]),
    )

    for X_train, X_test, y_train, y_test in config.X_y_by_cv:
        assert (
            len(X_train) == 0
        ), f"Expected X_train to be empty since there is only one fold, but got {len(X_train)}"
        assert (
            len(X_test) == 10
        ), f"Expected X_test to have 10 rows since there is only one fold, but got {len(X_test)}"
        assert (
            len(y_train) == 0
        ), f"Expected y_train to be empty since there is only one fold, but got {len(y_train)}"
        assert (
            len(y_test) == 10
        ), f"Expected y_test to have 10 elements since there is only one fold, but got {len(y_test)}"


def test_invalid_column_name():
    """Test with invalid column names."""
    invalid_column_data = {
        "feature1": range(10),
        "feature2": range(10, 20),
        "target": [1 if i % 2 == 0 else 0 for i in range(10)],
        "cv": [0 for _ in range(10)],
    }
    df_invalid = pl.DataFrame(invalid_column_data).lazy()
    with pytest.raises(KeyError) as e:
        UnivariateConfig(
            model_name="invalid_column_test",
            df_train=df_invalid,
            df_val_=df_invalid,
            target_column_name="invalid_target",
            feature_column_names=["feature1", "feature2"],
            time_series_validation=False,
            cv_folds=pl.Series("cv_folds", [0]),
        )

    assert "invalid_target" in str(
        e.value
    ), f"Expected 'invalid_target' in the error message, but got {e.value}"
