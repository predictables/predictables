# trunk-ignore-all(bandit)

from datetime import datetime
from typing import Literal

import numpy as np
import pandas as pd
import polars as pl
import pytest
from pandas.core.frame import DataFrame
from scipy.stats import pearsonr

from predictables.corr.src.pearson import pearson, pearson_expression

# set up logger
from predictables.util import Logger, to_pl_lf
from predictables.util.enums import DataType

logger = Logger(__name__).get_logger()


@pytest.fixture
def df():
    return pd.DataFrame(
        {
            "a": [1, 2, 4, 6, 7],
            "b": [2, 4, 6, 8, 10],
            "c": [1, 5, 6, 8, 13],
            "d": [20, 19, 16, 13, 4],
        }
    )


@pytest.fixture
def pl_df(df: DataFrame):
    return to_pl_lf(df)


@pytest.fixture
def invalid_df():
    df = pd.DataFrame(
        {
            "a": [0, 1, 0, 0, 1],
            "b": ["cat", "dog", "cat", "cat", "dog"],
            "c": [
                datetime(2020, 1, 1),
                datetime(2020, 1, 2),
                datetime(2020, 1, 3),
                datetime(2020, 1, 4),
                datetime(2020, 1, 5),
            ],
            "d": [20, 19, 16, 13, 4],
            "e": [True, False, True, False, True],
        }
    )
    df = to_pl_lf(df)
    return df.select(
        [
            pl.col("a").cast(pl.Int64).name.keep(),
            pl.col("b").cast(pl.Utf8).name.keep(),
            pl.col("c").cast(pl.Date).name.keep(),
            pl.col("d").cast(pl.Float64).name.keep(),
            pl.col("e").cast(pl.Boolean).name.keep(),
        ]
    )


@pytest.mark.parametrize(
    "column1,column2,dtype1,dtype2,expected",
    [
        ("a", "b", DataType.CONTINUOUS, DataType.CONTINUOUS, "correlation_a_b"),
        ("a", "a", DataType.CONTINUOUS, DataType.CONTINUOUS, "correlation_a_a"),
        ("a", "c", DataType.CONTINUOUS, DataType.CONTINUOUS, "correlation_a_c"),
        ("a", "d", DataType.CONTINUOUS, DataType.CONTINUOUS, "correlation_a_d"),
        ("b", "c", DataType.CONTINUOUS, DataType.CONTINUOUS, "correlation_b_c"),
        ("b", "d", DataType.CONTINUOUS, DataType.CONTINUOUS, "correlation_b_d"),
        ("c", "d", DataType.CONTINUOUS, DataType.CONTINUOUS, "correlation_c_d"),
    ],
)
def test_pearson_expression(
    pl_df: pl.LazyFrame,
    column1: Literal["a", "b", "c"],
    column2: Literal["b", "a", "c", "d"],
    dtype1: Literal[DataType.CONTINUOUS, DataType.CATEGORICAL],
    dtype2: Literal[DataType.CONTINUOUS, DataType.CATEGORICAL],
    expected: Literal[
        "correlation_a_b",
        "correlation_a_a",
        "correlation_a_c",
        "correlation_a_d",
        "correlation_b_c",
        "correlation_b_d",
        "correlation_c_d",
    ]
    | None,
):
    result = pl_df.select(pearson_expression(column1, column2, dtype1, dtype2))
    logger.debug(f"columns: {column1}, {column2}")
    logger.debug(f"dtype: {dtype1}, {dtype2}")
    logger.debug(f"expr: {pearson_expression(column1, column2, dtype1, dtype2)}")
    logger.debug(f"result.collect(): {result.collect()}")
    logger.debug(f"result col name: {result.columns[0]}")
    if result is not None:
        corr, _ = pearsonr(
            pl_df.select([column1]).collect()[column1],
            pl_df.select([column2]).collect()[column2],
        )
        logger.debug(f"corr: {corr}")
        assert (
            result.columns[0] == expected
        ), f"Expected the column name {expected}, but got {result.columns[0]}"
        assert np.round(result.collect().to_pandas()[expected][0], 3) == np.round(
            corr, 3
        ), f"Expected the correlation coefficient {np.round(corr, 3)}, but got {np.round(result.collect().to_pandas()[expected][0], 3)}"
    else:
        assert expected is None


def test_pearson(df: DataFrame):
    from itertools import product

    # Convert the test DataFrame to a Polars DataFrame
    pl_df = to_pl_lf(df)

    # Call the pearson function
    result = pearson(pl_df)

    # Assert that the result is a Polars DataFrame
    assert isinstance(result, pl.DataFrame)

    # Calculate the expected number of correlation pairs
    num_cols = len(df.columns)
    expected_pairs = num_cols * (num_cols - 1) // 2  # including self-correlation

    # Assert that the result has the correct shape
    assert result.shape[0] == 1  # Correlation results are in a single row
    assert result.shape[1] == expected_pairs

    # Check the correctness of the correlation values
    for col1, col2 in product(df.columns, df.columns):
        if col1 >= col2:  # Skip redundant calculations
            continue
        expected_corr, _ = pearsonr(df[col1], df[col2])
        actual_corr = result[f"correlation_{col1}_{col2}"][0]
        # Assert that each correlation coefficient is correct within a tolerance
        assert np.isclose(actual_corr, expected_corr, atol=1e-5)


@pytest.mark.parametrize(
    "column1,column2,dtype1,dtype2,expected",
    [
        ("a", "b", DataType.CATEGORICAL, DataType.CATEGORICAL, None),
        ("a", "c", DataType.CATEGORICAL, DataType.DATE, None),
        ("a", "d", DataType.CATEGORICAL, DataType.CONTINUOUS, None),
        ("a", "e", DataType.CATEGORICAL, DataType.BINARY, None),
        ("b", "c", DataType.CATEGORICAL, DataType.DATE, None),
        ("b", "d", DataType.CATEGORICAL, DataType.CONTINUOUS, None),
        ("b", "e", DataType.CATEGORICAL, DataType.BINARY, None),
        ("c", "d", DataType.DATE, DataType.CONTINUOUS, None),
        ("c", "e", DataType.DATE, DataType.BINARY, None),
        ("d", "e", DataType.CONTINUOUS, DataType.BINARY, None),
    ],
)
def test_pearson_expression_invalid_df(
    invalid_df: pl.LazyFrame,
    column1: Literal["a", "b", "c", "d"],
    column2: Literal["b", "c", "d", "e"],
    dtype1: Literal[
        DataType.CATEGORICAL, DataType.DATE, DataType.CONTINUOUS, DataType.BINARY
    ],
    dtype2: Literal[
        DataType.CATEGORICAL, DataType.DATE, DataType.CONTINUOUS, DataType.BINARY
    ],
    expected: None,
):
    if column1 != column2:
        df = invalid_df.select([pl.col(column1), pl.col(column2)]).collect()
        expr = pearson_expression(column1, column2, dtype1, dtype2)
        if expected is not None:
            assert (expr == expected) if expr is not None else False
        else:
            assert expr is None

        result = pearson(df.lazy())
        logger.debug(f"result: {result}")

        assert (
            result.shape[0] == 0
        ), f"result.shape[0] should be 0, but is {result.shape[0]}"
        assert (
            result.shape[1] == 0
        ), f"result.shape[1] should be 0, but is {result.shape[1]}"
