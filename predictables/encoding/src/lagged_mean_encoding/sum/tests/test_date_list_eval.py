import pytest
import polars as pl
import polars.testing as pltest
import datetime
from predictables.encoding.src.lagged_mean_encoding.sum.date_list_eval import (
    _date_list_eval,
    date_list_eval,
    _handle_date_list,
)
from predictables.encoding.src.lagged_mean_encoding.sum.get_date_list_col import (
    _get_date_list_col,
)
from predictables.encoding.src.lagged_mean_encoding.sum.get_value_map import (
    _get_value_map,
)


# Fixture for setting up a basic Polars dataframe with a list of dates
@pytest.fixture
def basic_date_df():
    return pl.DataFrame(
        {
            "date": [
                datetime.date(2022, 1, 1),
                datetime.date(2022, 1, 2),
                datetime.date(2022, 1, 3),
                datetime.date(2022, 1, 4),
                datetime.date(2022, 1, 5),
            ]
        }
    )


# Fixture for setting up the value map
@pytest.fixture
def value_map():
    return {
        datetime.date(2022, 1, 1): 1,
        datetime.date(2022, 1, 2): 2,
        datetime.date(2022, 1, 3): 3,
        datetime.date(2022, 1, 4): 4,
        datetime.date(2022, 1, 5): 5,
    }


@pytest.mark.parametrize(
    "window,expected",
    [(1, [1, 2, 3, 4, 5]), (2, [1, 3, 5, 7, 9]), (3, [1, 3, 6, 9, 12])],
)
def test__date_list_eval_handle_date_list(basic_date_df, window, expected):
    lf = _get_date_list_col(basic_date_df.lazy(), "date", 0, window).with_columns(
        [pl.col("date").dt.day().alias("x")]
    )
    lf = _get_date_list_col(lf, "date", 0, window)
    lf = _handle_date_list(lf.with_row_index(), "x", "date", "index")
    assert lf.collect() is not None
    pltest.assert_series_equal(
        lf.select("rolling_x").collect()["rolling_x"],
        pl.Series(expected),
        check_names=False,
    )
