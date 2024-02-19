import pandas as pd
import polars as pl
import numpy as np
import pytest
from predictables.util.src import _cv_filter as cvf


@pytest.fixture
def pd_cv_fold_data_ts():
    return pd.Series([0, 0, 1, 2, 3, 4, 5])


@pytest.fixture
def pd_cv_fold_data_no_ts():
    return pd.Series([1, 2, 3, 4, 5, 6, 7])


@pytest.fixture
def pd_series():
    return pd.Series(range(7))


@pytest.mark.parametrize(
    "f, train_test, expected",
    [
        (1, "train", pd.Series([0, 0])),
        (1, "test", pd.Series([1])),
        (2, "train", pd.Series([0, 0, 1])),
        (2, "test", pd.Series([2])),
        (3, "train", pd.Series([0, 0, 1, 2])),
        (3, "test", pd.Series([3])),
        (4, "train", pd.Series([0, 0, 1, 2, 3])),
        (4, "test", pd.Series([4])),
        (5, "train", pd.Series([0, 0, 1, 2, 3, 4])),
        (5, "test", pd.Series([5])),
    ],
)
@pytest.mark.parametrize("dtype", ["pandas", "polars", "numpy"])
@pytest.mark.parametrize("dtype2", ["pandas", "polars", "numpy"])
def test_filter_by_cv_fold_ts(
    pd_cv_fold_data_ts, pd_series, f, train_test, expected, dtype, dtype2
):
    if dtype == "pandas":
        s = pd_series
    elif dtype == "polars":
        s = pl.from_pandas(pd_series)
    elif dtype == "numpy":
        s = pd_series.values

    if dtype2 == "pandas":
        folds = pd_cv_fold_data_ts
    elif dtype2 == "polars":
        folds = pl.from_pandas(pd_cv_fold_data_ts)
    elif dtype2 == "numpy":
        folds = pd_cv_fold_data_ts.values

    result = cvf.filter_by_cv_fold(
        s, f, folds, time_series_validation=True, train_test=train_test
    )
    assert isinstance(
        result, pd.Series
    ), f"result is {type(result)}, expected pd.Series"
    # assert result.reset_index(drop=True).equals(
    #     expected.reset_index(drop=True)
    # ), f"result is {result.reset_index(drop=True)}, expected {expected.reset_index(drop=True)}"

    pd.testing.assert_series_equal(
        result.reset_index(drop=True), expected.reset_index(drop=True)
    )


@pytest.mark.parametrize(
    "f, train_test, expected",
    [
        (1, "train", pd.Series([2, 3, 4, 5, 6, 7])),
        (1, "test", pd.Series([1])),
        (2, "train", pd.Series([1, 3, 4, 5, 6, 7])),
        (2, "test", pd.Series([2])),
        (3, "train", pd.Series([1, 2, 4, 5, 6, 7])),
        (3, "test", pd.Series([3])),
        (4, "train", pd.Series([1, 2, 3, 5, 6, 7])),
        (4, "test", pd.Series([4])),
        (5, "train", pd.Series([1, 2, 3, 4, 6, 7])),
        (5, "test", pd.Series([5])),
        (6, "train", pd.Series([1, 2, 3, 4, 5, 7])),
        (6, "test", pd.Series([6])),
        (7, "train", pd.Series([1, 2, 3, 4, 5, 6])),
        (7, "test", pd.Series([7])),
    ],
)
@pytest.mark.parametrize("dtype", ["pandas", "polars", "numpy"])
@pytest.mark.parametrize("dtype2", ["pandas", "polars", "numpy"])
def test_filter_by_cv_fold_no_ts(
    pd_cv_fold_data_no_ts, pd_series, f, train_test, expected, dtype, dtype2
):
    if dtype == "pandas":
        s = pd_series
    elif dtype == "polars":
        s = pl.from_pandas(pd_series)
    elif dtype == "numpy":
        s = pd_series.values

    if dtype2 == "pandas":
        folds = pd_cv_fold_data_no_ts
    elif dtype2 == "polars":
        folds = pl.from_pandas(pd_cv_fold_data_no_ts)
    elif dtype2 == "numpy":
        folds = pd_cv_fold_data_no_ts.values

    result = cvf.filter_by_cv_fold(
        s, f, folds, time_series_validation=False, train_test=train_test
    ).reset_index(drop=True)
    assert isinstance(
        result.reset_index(drop=True), pd.Series
    ), f"result is {type(result)}, expected pd.Series"
    # assert result.reset_index(drop=True).equals(
    #     expected
    # ), f"result is {result.reset_index(drop=True)}, expected {expected.reset_index(drop=True)}"

    pd.testing.assert_series_equal(
        result.reset_index(drop=True), expected.reset_index(drop=True)
    )
