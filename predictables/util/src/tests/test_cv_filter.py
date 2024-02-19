# import numpy as np
# import pandas as pd
# import polars as pl
# import pytest

# from predictables.util.src._cv_filter import cv_filter
# from predictables.util.src._to_pd import to_pd_s


# @pytest.fixture
# def folds_pd():
#     return pd.Series(
#         ([0] * 20) + ([1] * 20) + ([2] * 20) + ([3] * 20) + ([4] * 20) + ([5] * 20)
#     )


# @pytest.fixture
# def folds_pl(folds_pd):
#     return pl.from_pandas(folds_pd)


# @pytest.fixture
# def folds_np(folds_pd):
#     return folds_pd.values


# @pytest.fixture
# def get_folds(folds_pd, folds_pl, folds_np):
#     return dict(pd_s=folds_pd, pl_s=folds_pl, np_s=folds_np)


# @pytest.mark.parametrize("fold", [1, 2, 3, 4, 5])
# @pytest.mark.parametrize("k", ["pd_s", "pl_s", "np_s"])
# def test_cv_filter_no_ts(get_folds, fold, k):
#     folds = get_folds[k]
#     expected = (
#         np.logical_not(np.equal(folds, fold))
#         if isinstance(folds, np.ndarray)
#         else folds != fold
#     )
#     result = cv_filter(fold, folds, ts_cv=False)
#     f = [r == e for r, e in zip(result.to_numpy(), expected)]
#     assert all(f), (
#         f"Non-time series: Failed for {k} with fold " f"{fold}: {result} != {expected}"
#     )
#     assert result.dtype == to_pd_s(expected).dtype, (
#         f"Non-time series: Failed for {k} with fold "
#         f"{fold}: {result.dtype} != {to_pd_s(expected).dtype}"
#     )


# @pytest.mark.parametrize("fold", [0, 1, 2, 3, 4, 5])
# @pytest.mark.parametrize("k", ["pd_s", "pl_s", "np_s"])
# def test_cv_filter_ts(get_folds, fold, k):
#     folds = get_folds[k]
#     expected = np.less(folds, fold) if isinstance(folds, np.ndarray) else folds < fold
#     result = cv_filter(fold, folds, ts_cv=True)
#     f = [r == e for r, e in zip(result.to_numpy(), expected)]
#     assert all(f), (
#         f"Time series: Failed for {k} " f"with fold {fold}: {result} != {expected}"
#     )
#     assert result.dtype == to_pd_s(expected).dtype, (
#         f"Time series: Failed for {k} "
#         f"with fold {fold}: {result.dtype} "
#         f"!= {to_pd_s(expected).dtype}"
#     )


# def test_cv_filter_with_empty_series():
#     empty_series = pd.Series([])
#     fold = 1  # Arbitrary fold number since series is empty
#     result_ts = cv_filter(fold, empty_series, ts_cv=True)
#     result_no_ts = cv_filter(fold, empty_series, ts_cv=False)

#     assert (
#         len(result_ts) == 0
#     ), f"Expected empty series for time series CV with empty input, but {result_ts} was returned (len={len(result_ts)})"
#     assert (
#         len(result_no_ts) == 0
#     ), f"Expected empty series for non-time series CV with empty input, but {result_no_ts} was returned (len={len(result_no_ts)})"


# @pytest.mark.parametrize("fold", [0, 1])
# def test_cv_filter_with_single_element_series(fold):
#     single_element_series = pd.Series([fold])
#     result_train_ts = cv_filter(
#         fold, single_element_series, ts_cv=True, train_test="train"
#     )
#     result_test_ts = cv_filter(
#         fold, single_element_series, ts_cv=True, train_test="test"
#     )

#     # For time series CV, the single element can only be part of training for fold > 0
#     assert result_train_ts.equals(
#         pd.Series([fold == 0])
#     ), f"Mismatch in time series CV train set for single-element series: result={result_train_ts}, expected={pd.Series([fold == 0])}"
#     assert result_test_ts.equals(
#         pd.Series([True])
#     ), f"Mismatch in time series CV test set for single-element series: result={result_test_ts}, expected={pd.Series([True])}"


# @pytest.mark.parametrize(
#     "invalid_fold", [-1, 6]
# )  # Assuming fold numbers in [0, 5] range
# def test_cv_filter_with_invalid_fold_numbers(get_folds, invalid_fold, k="pd_s"):
#     folds = get_folds[k]
#     # Assuming default behavior for invalid folds is an empty series or all False
#     result_ts = cv_filter(invalid_fold, folds, ts_cv=True)
#     result_no_ts = cv_filter(invalid_fold, folds, ts_cv=False)

#     assert (
#         not result_ts.any()
#     ), f"Expected no True values for invalid fold in time series CV, but got {result_ts.sum()} True values"
#     assert (
#         not result_no_ts.any()
#     ), f"Expected no True values for invalid fold in non-time series CV, but got {result_no_ts.sum()} True values"


# @pytest.mark.parametrize("fold", [0, 1, 2])
# def test_cv_filter_train_test_parameter_logic(get_folds, fold, k="pd_s"):
#     folds = get_folds[k]
#     # Time series CV, testing training and testing split explicitly
#     result_train = cv_filter(fold, folds, ts_cv=True, train_test="train")
#     result_test = cv_filter(fold, folds, ts_cv=True, train_test="test")

#     assert not result_train.equals(
#         result_test
#     ), f"Training and testing series should not match, but both are {result_train}"
#     assert result_train.sum() + result_test.sum() == len(
#         folds
#     ), f"Combined training and testing series should cover all data, but only {result_train.sum() + result_test.sum()} elements are covered, out of {len(folds)}"


# def test_cv_filter_unsupported_train_test_value():
#     with pytest.raises(ValueError) as e:
#         _ = cv_filter(1, pd.Series([0, 1, 2]), train_test="unsupported_value")
#     assert "Invalid value for train_test parameter" in str(
#         e.value
#     ), f"Expected ValueError for unsupported value for train_test parameter, but got {e.value}"
