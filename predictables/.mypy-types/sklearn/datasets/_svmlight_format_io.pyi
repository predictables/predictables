from ..utils import IS_PYPY as IS_PYPY, check_array as check_array
from ..utils._param_validation import (
    HasMethods as HasMethods,
    Interval as Interval,
    StrOptions as StrOptions,
    validate_params as validate_params,
)
from typing import Any

def load_svmlight_file(
    f,
    *,
    n_features: Any | None = ...,
    dtype=...,
    multilabel: bool = ...,
    zero_based: str = ...,
    query_id: bool = ...,
    offset: int = ...,
    length: int = ...
): ...
def load_svmlight_files(
    files,
    *,
    n_features: Any | None = ...,
    dtype=...,
    multilabel: bool = ...,
    zero_based: str = ...,
    query_id: bool = ...,
    offset: int = ...,
    length: int = ...
): ...
def dump_svmlight_file(
    X,
    y,
    f,
    *,
    zero_based: bool = ...,
    comment: Any | None = ...,
    query_id: Any | None = ...,
    multilabel: bool = ...
) -> None: ...
