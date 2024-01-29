from typing import Any

def inplace_csr_column_scale(X, scale) -> None: ...
def inplace_csr_row_scale(X, scale) -> None: ...
def mean_variance_axis(
    X, axis, weights: Any | None = ..., return_sum_weights: bool = ...
): ...
def incr_mean_variance_axis(
    X, *, axis, last_mean, last_var, last_n, weights: Any | None = ...
): ...
def inplace_column_scale(X, scale) -> None: ...
def inplace_row_scale(X, scale) -> None: ...
def inplace_swap_row_csc(X, m, n) -> None: ...
def inplace_swap_row_csr(X, m, n) -> None: ...
def inplace_swap_row(X, m, n) -> None: ...
def inplace_swap_column(X, m, n) -> None: ...
def min_max_axis(X, axis, ignore_nan: bool = ...): ...
def count_nonzero(X, axis: Any | None = ..., sample_weight: Any | None = ...): ...
def csc_median_axis_0(X): ...
