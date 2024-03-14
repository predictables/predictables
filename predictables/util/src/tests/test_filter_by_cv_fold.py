import pandas as pd  # type: ignore
import polars as pl
import polars.testing as pltest
import pytest

from predictables.util.src import _cv_filter as cvf
from predictables.util.src import _to_pl

to_pl_s = _to_pl.to_pl_s


@pytest.fixture
def pd_cv_fold_data_ts():
    return pd.Series([0, 0, 1, 2, 3, 4, 5])


@pytest.fixture
def pd_cv_fold_data_no_ts():
    return pd.Series([1, 2, 3, 4, 5, 6, 7])


@pytest.fixture
def pd_series():
    return pd.Series([i * 10 for i in range(7)])


@pytest.mark.parametrize(
    "f, train_test, expected",
    [
        (1, "train", pd.Series([0, 10])),
        (1, "test", pd.Series([20])),
        (2, "train", pd.Series([0, 10, 20])),
        (2, "test", pd.Series([30])),
        (3, "train", pd.Series([0, 10, 20, 30])),
        (3, "test", pd.Series([40])),
        (4, "train", pd.Series([0, 10, 20, 30, 40])),
        (4, "test", pd.Series([50])),
        (5, "train", pd.Series([0, 10, 20, 30, 40, 50])),
        (5, "test", pd.Series([60])),
    ],
)
@pytest.mark.parametrize("dtype", ["pandas", "polars", "numpy"])
@pytest.mark.parametrize("dtype2", ["pandas", "polars", "numpy"])
@pytest.mark.parametrize("output_type", ["pandas", "polars"])
def test_filter_by_cv_fold_ts(
    pd_cv_fold_data_ts, pd_series, f, train_test, expected, dtype, dtype2, output_type
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
        s,
        f,
        folds,
        time_series_validation=True,
        train_test=train_test,
        return_type=output_type,
    )
    if output_type == "pandas":
        result = result.reset_index(drop=True)
        result.name = None
    expected_output_type = pd.Series if output_type == "pandas" else pl.Series
    assert isinstance(
        result, expected_output_type
    ), f"result is {type(result)}, expected {expected_output_type}"

    if output_type == "pandas":
        pd.testing.assert_series_equal(
            result.reset_index(drop=True), expected.reset_index(drop=True)
        )
    else:
        expected_pl = to_pl_s(expected)
        pltest.assert_series_equal(result, expected_pl)


@pytest.mark.parametrize(
    "f, train_test, expected",
    [
        (1, "train", pd.Series([10, 20, 30, 40, 50, 60])),
        (1, "test", pd.Series([0])),
        (2, "train", pd.Series([0, 20, 30, 40, 50, 60])),
        (2, "test", pd.Series([10])),
        (3, "train", pd.Series([0, 10, 30, 40, 50, 60])),
        (3, "test", pd.Series([20])),
        (4, "train", pd.Series([0, 10, 20, 40, 50, 60])),
        (4, "test", pd.Series([30])),
        (5, "train", pd.Series([0, 10, 20, 30, 50, 60])),
        (5, "test", pd.Series([40])),
        (6, "train", pd.Series([0, 10, 20, 30, 40, 60])),
        (6, "test", pd.Series([50])),
        (7, "train", pd.Series([0, 10, 20, 30, 40, 50])),
        (7, "test", pd.Series([60])),
    ],
)
@pytest.mark.parametrize("dtype", ["pandas", "polars", "numpy"])
@pytest.mark.parametrize("dtype2", ["pandas", "polars", "numpy"])
@pytest.mark.parametrize("output_type", ["pandas", "polars"])
def test_filter_by_cv_fold_no_ts(
    pd_cv_fold_data_no_ts,
    pd_series,
    f,
    train_test,
    expected,
    dtype,
    dtype2,
    output_type,
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
        s,
        f,
        folds,
        time_series_validation=False,
        train_test=train_test,
        return_type=output_type,
    )

    if output_type == "pandas":
        result.name = None

    expected_output_type = pd.Series if output_type == "pandas" else pl.Series
    assert isinstance(
        result, expected_output_type
    ), f"result is {type(result)}, expected {expected_output_type}"

    if output_type == "pandas":
        pd.testing.assert_series_equal(
            result.reset_index(drop=True), expected.reset_index(drop=True)
        )
    else:
        expected_pl = to_pl_s(expected)
        pltest.assert_series_equal(result, expected_pl)
