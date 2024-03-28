# FILEPATH: /app/predictables/encoding/src/lagged_mean_encoding/sum/test_get_date_list_col.py

import datetime
import polars as pl
import pandas as pd
import pytest
from predictables.encoding.src.lagged_mean_encoding.sum.get_date_list_col import (
    _get_date_list_col,
)


# Create a fixture for the LazyFrame
@pytest.fixture
def lf():
    return (
        pl.DataFrame(
            {
                "date": [
                    datetime.date(2020, 1, 1) + datetime.timedelta(days=i)
                    for i in range(31)
                ],
                "value": range(1, 32),
            }
        )
        .lazy()
        .with_columns(
            [
                pl.col("date").cast(pl.Date).name.keep(),
                pl.col("value").cast(pl.Float64).name.keep(),
            ]
        )
    )


@pytest.fixture
def pd_df(lf):
    return lf.collect().to_pandas()


# Create a fixture for the date column name
@pytest.fixture
def date_col():
    return "date"


# Create a fixture for the offset
@pytest.fixture(params=[0, 10, 30])
def offset(request):
    return request.param


# Create a fixture for the window
@pytest.fixture(params=[30, 60, 90])
def window(request):
    return request.param


# Test the _get_date_list_col function
def test_get_date_list_col(lf, date_col, offset, window):
    result = _get_date_list_col(lf, date_col, offset, window)

    # Check that the result is a LazyFrame
    assert isinstance(
        result, pl.LazyFrame
    ), f"Expected the result to be a LazyFrame, got {type(result)}"

    # Check that the result has a "date_list" column
    assert (
        "date_list" in result.columns
    ), f"Expected 'date_list' column, got {result.columns}"

    # Collect the result and check the "date_list" column
    result_df = result.collect()
    for _, _, dt_list in result_df.rows():
        # Check that the "date_list" column contains a list of dates

        assert isinstance(dt_list, list), f"Expected list, got {type(dt_list)}"
        assert all(
            isinstance(date, datetime.date) for date in dt_list
        ), f"Expected list of dates, got:\n{dt_list}"

        # Check that the "date_list" column contains the correct number of dates
        assert len(dt_list) == window, f"Expected {window} dates, got {len(dt_list)}"


def test_pandas_df(pd_df, date_col, offset, window):
    result = _get_date_list_col(pd_df, date_col, offset, window)

    # Check that the result is a LazyFrame
    assert isinstance(
        result, pl.LazyFrame
    ), f"Expected the result to be a LazyFrame, got {type(result)}"

    # Check that the result has a "date_list" column
    assert (
        "date_list" in result.columns
    ), f"Expected 'date_list' column, got {result.columns}"

    # Collect the result and check the "date_list" column
    result_df = result.collect()
    for _, _, dt_list in result_df.rows():
        # Check that the "date_list" column contains a list of dates
        assert isinstance(dt_list, list), f"Expected list, got {type(dt_list)}"
        assert all(
            isinstance(date, datetime.date) for date in dt_list
        ), f"Expected list of dates, got:\n{dt_list}"

        # Check that the "date_list" column contains the correct number of dates
        assert len(dt_list) == window, f"Expected {window} dates, got {len(dt_list)}"


def test_zero_window(lf, date_col, offset):
    """Ensure that a window of 0 raises a ValueError."""
    with pytest.raises(ValueError):
        _get_date_list_col(lf, date_col, offset, 0)


# Test edge cases
@pytest.mark.parametrize(
    "offset, window",
    [(-1, 30), (0, 30), (1000, 30), (30, -1), (30, 0), (30, 1000), (0, 10), (10, 0)],
)
def test_get_date_list_col_edge_cases(lf, date_col, offset, window):
    if offset < 0 or window <= 0:
        with pytest.raises(ValueError):
            _get_date_list_col(lf, date_col, offset, window)
    else:
        result = _get_date_list_col(lf, date_col, offset, window)
        result_df = result.collect()
        for _, _, dtlist in result_df.rows():
            assert len(dtlist) == window


# Test invalid input
def test_get_date_list_col_invalid_input(lf):
    with pytest.raises(ValueError):
        _get_date_list_col("not a LazyFrame", "date", 30, 30)
    with pytest.raises(ValueError):
        _get_date_list_col(lf, "not a date column", 30, 30)


# Test correctness of dates
def test_get_date_list_col_correctness_of_dates(lf, date_col, offset, window):
    result = _get_date_list_col(lf, date_col, offset, window)
    result_df = result.collect()
    for dt, _, dt_list in result_df.rows():
        expected_dates = pl.from_pandas(
            pd.date_range(
                start=dt
                - pd.Timedelta(days=offset)
                - pd.Timedelta(days=window)
                + pd.Timedelta(days=1),
                end=dt - pd.Timedelta(days=offset),
            )
            .to_series(name="date")
            .reset_index()
        ).select(pl.col("date").cast(pl.Date).name.keep())
        assert dt_list == expected_dates["date"].to_list()
