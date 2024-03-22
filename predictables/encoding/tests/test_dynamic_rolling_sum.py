# FILEPATH: /app/predictables/encoding/src/tests/test_dynamic_rolling_sum.py

import pytest
import polars as pl
from predictables.encoding.src.lagged_mean_encoding._dynamic_rolling_sum import min_date, max_date, n_dates

@pytest.fixture(params=[
    (["2022-01-01", "2022-01-02", "2022-01-03"], "2022-01-01"),
    (["2022-01-03", "2022-01-02", "2022-01-01"], "2022-01-01"),
    (["2022-01-02", "2022-01-01", "2022-01-03"], "2022-01-01"),
])
def date_list_and_min(request):
    date_list, min_date = request.param
    df = pl.DataFrame({
        "date_list": [pl.Series([date_list])]
    }).with_columns([
        pl.col("date_list").cast(pl.Utf8).cast(pl.Date).name.keep()
    ])
    return df, min_date

def test_min_date(date_list_and_min):
    df, expected_min_date = date_list_and_min
    result = df.select(min_date()).collect()
    assert result["min_date"][0] == expected_min_date, f"Expected min_date to be {expected_min_date}, but got {result['min_date'][0]}"