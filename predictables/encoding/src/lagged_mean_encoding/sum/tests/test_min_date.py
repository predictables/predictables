import pytest
import polars as pl
import polars.testing as pltest
import datetime

from predictables.encoding.src.lagged_mean_encoding.sum.min_date import min_date


@pytest.fixture
def lf():
    return (
        pl.scan_parquet("predictables/encoding/tests/ts_testing_df.parquet")
        .select([pl.col("30_days_prior"), pl.col("60_days_prior")])
        .select(
            [
                pl.col("60_days_prior").name.keep(),
                pl.date_ranges(
                    pl.col("60_days_prior"), pl.col("30_days_prior"), "1d"
                ).alias("date_list"),
            ]
        )
    )


@pytest.fixture
def result(lf):
    return lf.select([min_date(), pl.col("60_days_prior"), pl.col("date_list")])


def test_min_date(lf):
    r = lf.select(min_date()).collect()["min_date"]
    e = lf.select(pl.col("60_days_prior").name.keep()).collect()["60_days_prior"]
    e2 = lf.select(pl.col("date_list").list.min().name.keep()).collect()["date_list"]

    assert r.shape == e.shape, f"Expected shape {e.shape}, got {r.shape}"
    pltest.assert_series_equal(r, e, check_names=False)
    pltest.assert_series_equal(r, e2, check_names=False)
