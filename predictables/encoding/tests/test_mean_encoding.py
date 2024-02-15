from datetime import date

import numpy as np
import polars as pl
import pytest

from predictables.encoding.src import _mean_encoding as me


# A fixture for reusable sample data
@pytest.fixture
def sample_data():
    if pl.__version__ >= "0.19.12":
        df = pl.DataFrame(
            {
                "category": ["A", "A", "B", "B", "A", "B"],
                "hit_count": [2, 3, 1, 4, 5, 6],
                "quote_count": [10, 12, 8, 10, 15, 20],
                "date": pl.date_range(
                    start=date(2022, 1, 1),
                    end=date(2022, 1, 6),
                    interval="1d",
                    eager=True,
                ),
            }
        ).with_columns(pl.col("category").cast(pl.Categorical).name.keep())
    else:
        df = pl.DataFrame(
            {
                "category": ["A", "A", "B", "B", "A", "B"],
                "hit_count": [2, 3, 1, 4, 5, 6],
                "quote_count": [10, 12, 8, 10, 15, 20],
                "date": pl.date_range(
                    start=date(2022, 1, 1),
                    end=date(2022, 1, 6),
                    interval="1d",
                    eager=True,
                ),
            }
        ).with_columns(pl.col("category").cast(pl.Categorical).keep_name())

    return df


def test_mean_encoding_hit_ratio1(sample_data):
    # Apply the function
    result_df = me.mean_encoding_with_ratio_lazy(
        sample_data,
        "category",
        "hit_count",
        "quote_count",
        "date",
        drop_mean_ratio=False,
    )
    result_df = (
        result_df.collect() if isinstance(result_df, pl.LazyFrame) else result_df
    )

    # Check if mean_ratio is calculated correctly for category 'A'
    # For the second 'A' data point:
    #   - Only the first 'A' should be used to calculate mean_ratio
    #   - Expected mean_ratio = policies_written / quotes_given = 2 / 10 = 0.2
    assert (
        result_df.filter(
            (pl.col("category") == "A") & (pl.col("date") == date(2022, 1, 2))
        )
        .select("mean_ratio")
        .to_numpy()[0][0]
        == 0.2
    )


def test_mean_encoding_hit_ratio2(sample_data):
    # Apply the function
    result_df = me.mean_encoding_with_ratio_lazy(
        sample_data,
        "category",
        "hit_count",
        "quote_count",
        "date",
        drop_mean_ratio=False,
    )
    result_df = (
        result_df.collect() if isinstance(result_df, pl.LazyFrame) else result_df
    )

    # Check if mean_ratio is calculated correctly for category 'B'
    # For the second 'B' data point:
    #   - Only the first 'B' should be used to calculate mean_ratio
    #   - Expected mean_ratio = policies_written / quotes_given = 1 / 8 = 0.125
    assert (
        result_df.filter(
            (pl.col("category") == "B") & (pl.col("date") == date(2022, 1, 4))
        )
        .select("mean_ratio")
        .to_numpy()[0][0]
        == 0.125
    )


def test_mean_encoding_hit_ratio3(sample_data):
    # Apply the function
    result_df = me.mean_encoding_with_ratio_lazy(
        sample_data,
        "category",
        "hit_count",
        "quote_count",
        "date",
        drop_cols=False,
        drop_mean_ratio=False,
    )
    result_df = (
        result_df.collect() if isinstance(result_df, pl.LazyFrame) else result_df
    )

    # Check if row order is preserved after operations
    np.testing.assert_allclose(
        result_df.select("row_ord").to_series().to_numpy(), np.array([0, 1, 2, 3, 4, 5])
    )
