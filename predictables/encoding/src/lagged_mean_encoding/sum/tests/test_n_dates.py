import pytest
import polars as pl
import polars.testing as pltest
import datetime

from predictables.encoding.src.lagged_mean_encoding.sum.n_dates import n_dates


@pytest.fixture
def lf():
    return pl.scan_parquet("predictables/encoding/tests/ts_testing_df.parquet").select(
        [
            pl.date_ranges(
                pl.col("60_days_prior"), pl.col("30_days_prior"), "1d"
            ).alias("date_list")
        ]
    )


@pytest.fixture
def result(lf):
    return lf.select([n_dates(), pl.col("date_list")])


def test_n_dates(lf):
    r = lf.select(n_dates()).collect()["n_dates"]
    e = lf.select(pl.col("date_list").list.len().alias("n_dates")).collect()["n_dates"]

    assert r.shape == e.shape, f"Expected shape {e.shape}, got {r.shape}"
    pltest.assert_series_equal(r, e, check_names=False)
