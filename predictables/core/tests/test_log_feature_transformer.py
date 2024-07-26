import numpy as np
import pytest
import polars as pl
from predictables.core.src.univariate_feature_transformer import LogTransformer
from predictables.core.src.univariate_config import UnivariateConfig


@pytest.fixture
def log_transformer():
    train_data = pl.DataFrame(
        {
            "feature1": [1, 2, 3, 4, 5],
            # exponential function to ensure skewness:
            "feature2": np.exp([2, 3, 4, 5, 6]),
            "target": [0, 1, 0, 1, 0],
        }
    ).lazy()

    val_data = pl.DataFrame(
        {
            "feature1": [5, 4, 3, 2, 1],
            "feature2": np.exp([6, 5, 4, 3, 2]),
            "target": [1, 0, 1, 0, 1],
        }
    ).lazy()

    config = UnivariateConfig(
        model_name="test_model",
        df_train=train_data,
        df_val_=val_data,
        target_column_name="target",
        feature_column_names=["feature1", "feature2"],
    )

    return LogTransformer(config=config)


def test_log_transformer_initialization(log_transformer):
    """Test the initialization of the LogTransformer."""
    assert (
        log_transformer.config.model_name == "test_model"
    ), f"Expected model_name to be 'test_model', but got {log_transformer.config.model_name}"
    assert (
        log_transformer.df is not None
    ), f"Expected df to be not None, but got {log_transformer.df}"
    assert (
        log_transformer.df_val is not None
    ), f"Expected df_val to be not None, but got {log_transformer.df_val}"
    assert (
        log_transformer.feature_column_names == ["feature1", "feature2"]
    ), f"Expected feature_column_names to be ['feature1', 'feature2'], but got {log_transformer.feature_column_names}"


def test_log_transformer_skewness_threshold(log_transformer):
    """Test the skewness threshold property of the LogTransformer."""
    assert (
        log_transformer.skewness_threshold == 0.5
    ), f"Expected skewness_threshold to be 0.5, but got {log_transformer.skewness_threshold}"
    log_transformer.skewness_threshold = 0.6
    assert (
        log_transformer.skewness_threshold == 0.6
    ), f"Expected skewness_threshold to be 0.6, but got {log_transformer.skewness_threshold}"


def test_calculate_skewness(log_transformer):
    """Test the calculation of skewness for a feature."""
    skewness = log_transformer._calculate_skewness("feature1")  # noqa: SLF001
    assert isinstance(
        skewness, float
    ), f"Expected skewness to be a float, but got {skewness}"


@pytest.mark.parametrize("skewness, expected", [(0.6, True), (0.4, False)])
def test_should_apply_log_transform(log_transformer, skewness, expected):
    """Test the determination of whether log transformation should be applied."""
    assert (
        log_transformer._should_apply_log_transform(skewness) == expected  # noqa: SLF001
    ), f"Expected should_apply_log_transform to be {expected}, but got {log_transformer._should_apply_log_transform(skewness)}"  # noqa: SLF001


def test_apply_log_transform(log_transformer):
    log_transformer._apply_log_transform("feature1")  # noqa: SLF001
    df = log_transformer.df.collect()
    assert (
        "log1p_feature1" in df.columns
    ), f"Expected 'log1p_feature1' in columns, but got {df.columns}"


def test_transform_features(log_transformer):
    """Test the transformation of features using the LogTransformer."""
    # Make the skewness threshold lower to apply log transformation to
    # ensure that at least one feature is transformed
    log_transformer.skewness_threshold = 0.1
    transformed_features = log_transformer.transform_features()
    assert (
        "feature1" in transformed_features
    ), f"Expected 'feature1' in transformed features, but got {transformed_features}"
    assert (
        "feature2" in transformed_features
    ), f"Expected 'feature2' in transformed features, but got {transformed_features}"
    assert (
        "log1p_feature1" in transformed_features
        or "log1p_feature2" in transformed_features
    ), f"Expected 'log1p_feature1' or 'log1p_feature2' in transformed features, but got {transformed_features}"


def test_specific_skewness_threshold():
    """Test transformation where feature1 has skewness 0.1 and feature2 has skewness 1.0."""
    train_data = pl.DataFrame(
        {
            "feature1": [10, 10.5, 11, 11.5, 12],  # Low skewness
            "feature2": [1, 10, 100, 1000, 10000],  # High skewness
            "target": [0, 1, 0, 1, 0],
        }
    ).lazy()

    val_data = pl.DataFrame(
        {
            "feature1": [12, 11.5, 11, 10.5, 10],
            "feature2": [10000, 1000, 100, 10, 1],
            "target": [1, 0, 1, 0, 1],
        }
    ).lazy()

    config = UnivariateConfig(
        model_name="test_model",
        df_train=train_data,
        df_val_=val_data,
        target_column_name="target",
        feature_column_names=["feature1", "feature2"],
    )

    transformer = LogTransformer(config=config)
    transformer.skewness_threshold = 0.5  # Ensure threshold is 0.5

    transformed_features = transformer.transform_features()
    assert (
        "feature1" in transformed_features
    ), f"Expected 'feature1' in transformed features, but got {transformed_features}"
    assert (
        "feature2" in transformed_features
    ), f"Expected 'feature2' in transformed features, but got {transformed_features}"
    assert (
        "log1p_feature2" in transformed_features
    ), f"Expected 'log1p_feature2' in transformed features, but got {transformed_features}"
    assert (
        "log1p_feature1" not in transformed_features
    ), f"Did not expect 'log1p_feature1' in transformed features, but got {transformed_features}"
