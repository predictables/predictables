import pytest
import polars as pl
from polars.testing import assert_frame_equal
from predictables.core.src.univariate_feature_transformer import BaseFeatureTransformer
from predictables.core.src.univariate_config import UnivariateConfig


@pytest.fixture
def base_transformer():
    train_data = pl.DataFrame(
        {
            "feature1": [1, 2, 3, 4, 5],
            "feature2": [2, 3, 4, 5, 6],
            "target": [0, 1, 0, 1, 0],
        }
    ).lazy()

    val_data = pl.DataFrame(
        {
            "feature1": [5, 4, 3, 2, 1],
            "feature2": [6, 5, 4, 3, 2],
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

    return BaseFeatureTransformer(config=config)


def test_base_transformer_initialization(base_transformer):
    assert (
        base_transformer.config.model_name == "test_model"
    ), f"Expected model_name to be 'test_model', but got {base_transformer.config.model_name}"
    assert (
        base_transformer.df is not None
    ), f"Expected df to be not None, but got {base_transformer.df}"
    assert (
        base_transformer.df_val is not None
    ), f"Expected df_val to be not None, but got {base_transformer.df_val}"
    assert (
        base_transformer.feature_column_names == ["feature1", "feature2"]
    ), f"Expected feature_column_names to be ['feature1', 'feature2'], but got {base_transformer.feature_column_names}"


def test_base_transformer_df_setter(base_transformer):
    new_train_data = pl.DataFrame(
        {
            "feature1": [10, 20, 30, 40, 50],
            "feature2": [20, 30, 40, 50, 60],
            "target": [0, 1, 0, 1, 0],
        }
    ).lazy()

    base_transformer.df = new_train_data
    assert_frame_equal(base_transformer.df.collect(), new_train_data.collect())


def test_base_transformer_df_val_setter(base_transformer):
    new_val_data = pl.DataFrame(
        {
            "feature1": [50, 40, 30, 20, 10],
            "feature2": [60, 50, 40, 30, 20],
            "target": [1, 0, 1, 0, 1],
        }
    ).lazy()

    base_transformer.df_val = new_val_data
    assert_frame_equal(base_transformer.df_val.collect(), new_val_data.collect())
