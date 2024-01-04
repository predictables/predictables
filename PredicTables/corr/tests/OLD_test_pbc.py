# trunk-ignore-all(bandit)

import numpy as np
import pandas as pd
import polars as pl
import pytest
from pandas.core.frame import DataFrame
from scipy.stats import pointbiserialr

from PredicTables.corr.src.point_biserial import (
    point_biserial,
    point_biserial_expression,
)

# set up logger
from PredicTables.util import Logger, to_pl_lf
from PredicTables.util.enums import DataType

logger = Logger(__name__).get_logger()


@pytest.fixture
def continuous_binary_df():
    return pd.DataFrame(
        {
            "continuous": np.random.randn(
                100
            ),  # Replace with appropriate continuous data
            "binary": np.random.choice([0, 1], size=100),
            "non_binary": ["cat", "dog", "fish", "hampster"] * 25,
            "non_continuous": ["kitty", "doggy", "fishy", "hampsterboi"] * 25,
        }
    )


@pytest.fixture
def pl_continuous_binary_df(continuous_binary_df: DataFrame):
    return to_pl_lf(continuous_binary_df).with_columns(
        [
            pl.col("continuous").cast(pl.Float64).name.keep(),
            pl.col("binary").cast(pl.Int8).name.keep(),
            pl.col("non_binary").cast(pl.Utf8).name.keep(),
            pl.col("non_continuous").cast(pl.Utf8).name.keep(),
        ]
    )


@pytest.fixture
def invalid_df():
    return pd.DataFrame(
        {
            "continuous": np.random.randn(100) * 100,
            "non_binary": ["cat", "dog", "fish", "hampster"] * 25,
            "non_continuous": ["kitty", "doggy", "fishy", "hampsterboi"] * 25,
        }
    )


@pytest.mark.parametrize(
    "continuous_col,binary_col,expected",
    [
        ("continuous", "binary", True),
        ("continuous", "non_binary", False),
        ("non_continuous", "binary", False),
    ],
)
def test_point_biserial_expression(
    pl_continuous_binary_df: pl.LazyFrame,
    continuous_col: str,
    binary_col: str,
    expected: bool,
):
    dtype_continuous = DataType.CONTINUOUS
    dtype_binary = DataType.BINARY if binary_col == "binary" else DataType.CATEGORICAL

    logger.debug(f"continuous_col: {continuous_col}")
    logger.debug(f"binary_col: {binary_col}")
    logger.debug(f"dtype_continuous: {dtype_continuous}")
    logger.debug(f"dtype_binary: {dtype_binary}")

    expr = point_biserial_expression(
        continuous_col, binary_col, dtype_continuous, dtype_binary
    )
    logger.debug(f"expr is expected to not be None: {expr}")
    if expected:
        assert expr is not None, f"Expression is {expr}, expected {expected}"
    else:
        assert expr is None, f"Expression is {expr}, expected {expected}"


def test_point_biserial(continuous_binary_df: DataFrame):
    logger.debug(f"continuous_binary_df: {continuous_binary_df}")
    pl_df = to_pl_lf(continuous_binary_df)
    logger.debug(f"pl_df: {pl_df.collect()}")

    result = point_biserial(pl_df)
    logger.debug(f"result: {result}")
    logger.debug(f"type(result) (expect pl.DataFrame): {type(result)}")

    assert isinstance(result, pl.DataFrame)

    # Ensure binary column is correctly formatted as numeric for comparison
    continuous_binary_df["binary"] = continuous_binary_df["binary"].astype(int)
    logger.debug(f"continuous_binary_df: {continuous_binary_df}")

    expected_corr, _ = pointbiserialr(
        continuous_binary_df["binary"], continuous_binary_df["continuous"]
    )

    logger.debug(f"expected_corr: {expected_corr}")
    actual_corr = result[result.columns[0]][0]

    logger.debug(f"actual_corr: {actual_corr}")

    assert np.round(actual_corr, 3) == np.round(
        expected_corr, 3
    ), f"Correlation coefficients are not close:\nactual ({actual_corr}) != expected ({expected_corr})"


# @pytest.mark.parametrize(
#     "continuous_col,binary_col",
#     [
#         ("continuous", "non_binary"),
#         ("non_continuous", "binary"),
#     ],
# )
# def test_point_biserial_expression_invalid_df(
#     invalid_df: pl.LazyFrame,
#     continuous_col: str,
#     binary_col: str,
# ):
#     pl_df = to_pl_lf(invalid_df)
#     dtype_continuous = DataType.CONTINUOUS
#     dtype_binary = DataType.BINARY if binary_col == "binary" else DataType.CATEGORICAL

#     expr = point_biserial_expression(continuous_col, binary_col, dtype_continuous, dtype_binary)
#     assert expr is None

#     # Test the overall function behavior with invalid data types
#     result = point_biserial(pl_df)
#     assert result.shape[0] == 0 and result.shape[1] == 0
