import pytest
import polars as pl
import datetime

from predictables.encoding.src.lagged_mean_encoding.sum.get_value_map import (
    _get_value_map,
)


@pytest.fixture
def lf():
    """Test the functionality with a predefined lazyframe fixture."""
    return pl.DataFrame(
        {
            "date_col": [
                "2021-01-01",
                "2021-01-01",
                "2021-01-01",
                "2021-01-02",
                "2021-01-02",
                "2021-01-02",
                "2021-01-03",
                "2021-01-03",
                "2021-01-03",
            ],
            "red_herring_category_col": ["a", "b", "c", "a", "b", "c", "a", "b", "c"],
            "value_col": [1, 2, 3, 4, 5, 6, 7, 8, 9],
        }
    ).lazy()


@pytest.fixture
def expected():
    """Return the expected dictionary after formatting."""
    return {
        datetime.date(2021, 1, 1): 1.0,
        datetime.date(2021, 1, 2): 2.0,
        datetime.date(2021, 1, 3): 3.0,
    }
