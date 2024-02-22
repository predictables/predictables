import polars as pl
import pytest
from predictables.encoding.src.lagged_mean_encoding._calculate_dynamic_window import (
    calculate_dynamic_window,
)


@pytest.fixture
def df():
    data = {
        "date": [
            "2015-01-01",
            "2015-02-01",
            "2015-03-01",
            "2015-04-01",
            "2015-05-01",
            "2015-06-01",
            "2015-07-01",
            "2015-08-01",
            "2015-09-01",
            "2015-10-01",
            "2015-11-01",
            "2015-12-01",
            "2016-01-01",
            "2016-02-01",
            "2016-03-01",
            "2016-04-01",
            "2016-05-01",
            "2016-06-01",
            "2016-07-01",
            "2016-08-01",
            "2016-09-01",
            "2016-10-01",
            "2016-11-01",
            "2016-12-01",
            "2017-01-01",
            "2017-02-01",
            "2017-03-01",
            "2017-04-01",
            "2017-05-01",
            "2017-06-01",
            "2017-07-01",
            "2017-08-01",
            "2017-09-01",
            "2017-10-01",
            "2017-11-01",
            "2017-12-01",
        ],
        "num": [10.0 * i for i in range(36)],
        "denom": [100.0 * i for i in range(36)],
    }
    df = pl.DataFrame(data)
    df = df.with_columns(pl.col("date").str.strptime(pl.Date, "%Y-%m-%d"))
    return df.lazy()


@pytest.mark.parametrize(
    "year,month,expected_len",
    [
        (2017, 12, 12),
        (2017, 11, 12),
        (2017, 10, 12),
        (2017, 9, 12),
        (2017, 8, 12),
        (2017, 7, 12),
        (2017, 6, 12),
        # skip in between -- they are all length 12
        (2016, 5, 12),
        (2016, 4, 12),
        (2016, 3, 12),
        (2016, 2, 12),
        (2016, 1, 12),
        (2015, 12, 11),
        (2015, 11, 10),
        (2015, 10, 9),
        (2015, 9, 8),
        (2015, 8, 7),
        (2015, 7, 6),
        (2015, 6, 5),
        (2015, 5, 4),
        (2015, 4, 3),
        (2015, 3, 2),
        (2015, 2, 1),
        (2015, 1, 0),
    ],
)
def test_calculate_dynamic_window_no_gp(df, year, month, expected_len):
    """
    Tests that the filter_rolling_year_prior function filters the lazyframe to
    include the proper number of items -- only from the rolling year prior.

    When the data do not have 12 prior months, the function should return the
    maximum number of months available. The function should return an empty
    lazyframe when there are no prior months available (eg for the first date
    in the data).
    """

    result = calculate_dynamic_window(df, "date", "num", "denom")
    nrows = df.collect().select(result).shape[0]
    assert nrows == expected_len, f"Expected {expected_len} rows, got {nrows} rows."
