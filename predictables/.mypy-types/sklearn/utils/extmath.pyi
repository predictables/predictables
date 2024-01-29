from . import check_random_state as check_random_state
from ._array_api import get_namespace as get_namespace
from .sparsefuncs_fast import csr_row_norms as csr_row_norms
from .validation import check_array as check_array
from typing import Any

def squared_norm(x): ...
def row_norms(X, squared: bool = ...): ...
def fast_logdet(A): ...
def density(w, **kwargs): ...
def safe_sparse_dot(a, b, *, dense_output: bool = ...): ...
def randomized_range_finder(
    A,
    *,
    size,
    n_iter,
    power_iteration_normalizer: str = ...,
    random_state: Any | None = ...
): ...
def randomized_svd(
    M,
    n_components,
    *,
    n_oversamples: int = ...,
    n_iter: str = ...,
    power_iteration_normalizer: str = ...,
    transpose: str = ...,
    flip_sign: bool = ...,
    random_state: Any | None = ...,
    svd_lapack_driver: str = ...
): ...
def weighted_mode(a, w, *, axis: int = ...): ...
def cartesian(arrays, out: Any | None = ...): ...
def svd_flip(u, v, u_based_decision: bool = ...): ...
def log_logistic(X, out: Any | None = ...): ...
def softmax(X, copy: bool = ...): ...
def make_nonnegative(X, min_value: int = ...): ...
def stable_cumsum(
    arr, axis: Any | None = ..., rtol: float = ..., atol: float = ...
): ...
