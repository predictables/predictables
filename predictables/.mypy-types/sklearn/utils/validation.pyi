from ..exceptions import (
    DataConversionWarning as DataConversionWarning,
    NotFittedError as NotFittedError,
    PositiveSpectrumWarning as PositiveSpectrumWarning,
)
from ..utils._array_api import get_namespace as get_namespace
from ..utils.fixes import ComplexWarning as ComplexWarning
from ._isfinite import FiniteStatus as FiniteStatus, cy_isfinite as cy_isfinite
from typing import Any

FLOAT_DTYPES: Any

def assert_all_finite(
    X, *, allow_nan: bool = ..., estimator_name: Any | None = ..., input_name: str = ...
) -> None: ...
def as_float_array(X, *, copy: bool = ..., force_all_finite: bool = ...): ...
def check_memory(memory): ...
def check_consistent_length(*arrays) -> None: ...
def indexable(*iterables): ...
def check_array(
    array,
    accept_sparse: bool = ...,
    *,
    accept_large_sparse: bool = ...,
    dtype: str = ...,
    order: Any | None = ...,
    copy: bool = ...,
    force_all_finite: bool = ...,
    ensure_2d: bool = ...,
    allow_nd: bool = ...,
    ensure_min_samples: int = ...,
    ensure_min_features: int = ...,
    estimator: Any | None = ...,
    input_name: str = ...
): ...
def check_X_y(
    X,
    y,
    accept_sparse: bool = ...,
    *,
    accept_large_sparse: bool = ...,
    dtype: str = ...,
    order: Any | None = ...,
    copy: bool = ...,
    force_all_finite: bool = ...,
    ensure_2d: bool = ...,
    allow_nd: bool = ...,
    multi_output: bool = ...,
    ensure_min_samples: int = ...,
    ensure_min_features: int = ...,
    y_numeric: bool = ...,
    estimator: Any | None = ...
): ...
def column_or_1d(y, *, dtype: Any | None = ..., warn: bool = ...): ...
def check_random_state(seed): ...
def has_fit_parameter(estimator, parameter): ...
def check_symmetric(
    array, *, tol: float = ..., raise_warning: bool = ..., raise_exception: bool = ...
): ...
def check_is_fitted(
    estimator, attributes: Any | None = ..., *, msg: Any | None = ..., all_or_any=...
) -> None: ...
def check_non_negative(X, whom) -> None: ...
def check_scalar(
    x,
    name,
    target_type,
    *,
    min_val: Any | None = ...,
    max_val: Any | None = ...,
    include_boundaries: str = ...
): ...
