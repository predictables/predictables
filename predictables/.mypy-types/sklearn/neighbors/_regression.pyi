from ..base import RegressorMixin as RegressorMixin
from ..utils._param_validation import StrOptions as StrOptions
from ._base import (
    KNeighborsMixin as KNeighborsMixin,
    NeighborsBase as NeighborsBase,
    RadiusNeighborsMixin as RadiusNeighborsMixin,
)
from typing import Any

class KNeighborsRegressor(KNeighborsMixin, RegressorMixin, NeighborsBase):
    weights: Any
    def __init__(
        self,
        n_neighbors: int = ...,
        *,
        weights: str = ...,
        algorithm: str = ...,
        leaf_size: int = ...,
        p: int = ...,
        metric: str = ...,
        metric_params: Any | None = ...,
        n_jobs: Any | None = ...
    ) -> None: ...
    def fit(self, X, y): ...
    def predict(self, X): ...

class RadiusNeighborsRegressor(RadiusNeighborsMixin, RegressorMixin, NeighborsBase):
    weights: Any
    def __init__(
        self,
        radius: float = ...,
        *,
        weights: str = ...,
        algorithm: str = ...,
        leaf_size: int = ...,
        p: int = ...,
        metric: str = ...,
        metric_params: Any | None = ...,
        n_jobs: Any | None = ...
    ) -> None: ...
    def fit(self, X, y): ...
    def predict(self, X): ...
