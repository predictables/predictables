import numpy as np
import pandas as pd
import polars as pl
import pytest

from predictables.util.src._cv_filter import cv_filter
from predictables.util.src._to_pd import to_pd_s


@pytest.fixture
def folds_pd():
    return pd.Series(
        ([0] * 20) + ([1] * 20) + ([2] * 20) + ([3] * 20) + ([4] * 20) + ([5] * 20)
    )


@pytest.fixture
def folds_pl(folds_pd):
    return pl.from_pandas(folds_pd)


@pytest.fixture
def folds_np(folds_pd):
    return folds_pd.values


@pytest.fixture
def get_folds(folds_pd, folds_pl, folds_np):
    return dict(pd_s=folds_pd, pl_s=folds_pl, np_s=folds_np)


@pytest.mark.parametrize("fold", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("k", ["pd_s", "pl_s", "np_s"])
def test_cv_filter_no_ts(get_folds, fold, k):
    folds = get_folds[k]
    expected = np.logical_not(np.equal(folds, fold)) if isinstance(folds, np.ndarray) else folds != fold
    result = cv_filter(fold, folds, ts_cv=False)
    f = [r == e for r, e in zip(result.to_numpy(), expected)]
    assert all(f), (
        f"Non-time series: Failed for {k} with fold " f"{fold}: {result} != {expected}"
    )
    assert result.dtype == to_pd_s(expected).dtype, (
        f"Non-time series: Failed for {k} with fold "
        f"{fold}: {result.dtype} != {to_pd_s(expected).dtype}"
    )


@pytest.mark.parametrize("fold", [0, 1, 2, 3, 4, 5])
@pytest.mark.parametrize("k", ["pd_s", "pl_s", "np_s"])
def test_cv_filter_ts(get_folds, fold, k):
    folds = get_folds[k]
    expected = (
        np.less(folds, fold) if isinstance(folds, np.ndarray) else folds < fold
    )
    result = cv_filter(fold, folds, ts_cv=True)
    f = [r == e for r, e in zip(result.to_numpy(), expected)]
    assert all(f), (
        f"Time series: Failed for {k} " f"with fold {fold}: {result} != {expected}"
    )
    assert result.dtype == to_pd_s(expected).dtype, (
        f"Time series: Failed for {k} "
        f"with fold {fold}: {result.dtype} "
        f"!= {to_pd_s(expected).dtype}"
    )
