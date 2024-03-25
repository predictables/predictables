import pytest
import polars as pl
import polars.testing as pltest
import datetime

from predictables.encoding.src.lagged_mean_encoding.sum.format_columns import (
    _format_date_col,
    _format_value_col,
)


@pytest.fixture
def lf():
    """Test the functionality with a predefined lazyframe fixture."""
    return pl.DataFrame(
        {
            "date_col1": ["2021-01-01", "2021-01-02", "2021-01-03"],
            "date_col2": [
                datetime.datetime(2021, 1, 1),
                datetime.datetime(2021, 1, 2),
                datetime.datetime(2021, 1, 3),
            ],
            "date_col3": [
                datetime.date(2021, 1, 1),
                datetime.date(2021, 1, 2),
                datetime.date(2021, 1, 3),
            ],
            "value_col1": [1, 2, 3],
            "value_col2": [1.0, 2.0, 3.0],
        }
    ).lazy()


@pytest.fixture
def expected_lf():
    """Return the expected LazyFrame after formatting."""
    return pl.DataFrame(
        {
            "date_col": [
                datetime.date(2021, 1, 1),
                datetime.date(2021, 1, 2),
                datetime.date(2021, 1, 3),
            ],
            "value_col": [1.0, 2.0, 3.0],
        }
    ).lazy()


@pytest.mark.parametrize("date_col", ["date_col1", "date_col2", "date_col3"])
def test_format_date_col(lf, expected_lf, date_col):
    """Test that the _format_date_col function correctly formats the date column."""
    result = (
        _format_date_col(lf, date_col).select([pl.col(date_col)]).collect()[date_col]
    )
    expected = expected_lf.select([pl.col("date_col")]).collect()["date_col"]

    pltest.assert_series_equal(result, expected, check_names=False)


@pytest.mark.parametrize("value_col", ["value_col1", "value_col2"])
def test_format_value_col(lf, expected_lf, value_col):
    """Test that the _format_value_col function correctly formats the value column."""
    result = (
        _format_value_col(lf, value_col)
        .select([pl.col(value_col)])
        .collect()[value_col]
    )
    expected = expected_lf.select([pl.col("value_col")]).collect()["value_col"]

    pltest.assert_series_equal(result, expected, check_names=False)


def test_format_columns(lf, expected_lf):
    """Test that the format_columns function properly formats the date and value columns."""
    test_lf = lf.select(
        [pl.col("date_col1").alias("date_col"), pl.col("value_col1").alias("value_col")]
    )
    result = _format_date_col(test_lf, "date_col")
    result = _format_value_col(result, "value_col")
    expected = expected_lf
    pltest.assert_frame_equal(result, expected)