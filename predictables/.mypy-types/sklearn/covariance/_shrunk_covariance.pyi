from . import (
    EmpiricalCovariance as EmpiricalCovariance,
    empirical_covariance as empirical_covariance,
)
from ..utils import check_array as check_array
from ..utils._param_validation import (
    Interval as Interval,
    validate_params as validate_params,
)
from typing import Any

def shrunk_covariance(emp_cov, shrinkage: float = ...): ...

class ShrunkCovariance(EmpiricalCovariance):
    shrinkage: Any
    def __init__(
        self,
        *,
        store_precision: bool = ...,
        assume_centered: bool = ...,
        shrinkage: float = ...
    ) -> None: ...
    location_: Any
    def fit(self, X, y: Any | None = ...): ...

def ledoit_wolf_shrinkage(X, assume_centered: bool = ..., block_size: int = ...): ...
def ledoit_wolf(X, *, assume_centered: bool = ..., block_size: int = ...): ...

class LedoitWolf(EmpiricalCovariance):
    block_size: Any
    def __init__(
        self,
        *,
        store_precision: bool = ...,
        assume_centered: bool = ...,
        block_size: int = ...
    ) -> None: ...
    location_: Any
    shrinkage_: Any
    def fit(self, X, y: Any | None = ...): ...

def oas(X, *, assume_centered: bool = ...): ...

class OAS(EmpiricalCovariance):
    location_: Any
    shrinkage_: Any
    def fit(self, X, y: Any | None = ...): ...
