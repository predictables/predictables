from ... import get_config as get_config
from .._dist_metrics import (
    BOOL_METRICS as BOOL_METRICS,
    METRIC_MAPPING64 as METRIC_MAPPING64,
)
from ._argkmin import ArgKmin32 as ArgKmin32, ArgKmin64 as ArgKmin64
from ._argkmin_classmode import (
    ArgKminClassMode32 as ArgKminClassMode32,
    ArgKminClassMode64 as ArgKminClassMode64,
)
from ._radius_neighbors import (
    RadiusNeighbors32 as RadiusNeighbors32,
    RadiusNeighbors64 as RadiusNeighbors64,
)
from abc import abstractmethod
from typing import Any, List

def sqeuclidean_row_norms(X, num_threads): ...

class BaseDistancesReductionDispatcher:
    @classmethod
    def valid_metrics(cls) -> List[str]: ...
    @classmethod
    def is_usable_for(cls, X, Y, metric) -> bool: ...
    @classmethod
    @abstractmethod
    def compute(cls, X, Y, **kwargs): ...

class ArgKmin(BaseDistancesReductionDispatcher):
    @classmethod
    def compute(
        cls,
        X,
        Y,
        k,
        metric: str = ...,
        chunk_size: Any | None = ...,
        metric_kwargs: Any | None = ...,
        strategy: Any | None = ...,
        return_distance: bool = ...,
    ): ...

class RadiusNeighbors(BaseDistancesReductionDispatcher):
    @classmethod
    def compute(
        cls,
        X,
        Y,
        radius,
        metric: str = ...,
        chunk_size: Any | None = ...,
        metric_kwargs: Any | None = ...,
        strategy: Any | None = ...,
        return_distance: bool = ...,
        sort_results: bool = ...,
    ): ...

class ArgKminClassMode(BaseDistancesReductionDispatcher):
    @classmethod
    def is_usable_for(cls, X, Y, metric) -> bool: ...
    @classmethod
    def compute(
        cls,
        X,
        Y,
        k,
        weights,
        labels,
        unique_labels,
        metric: str = ...,
        chunk_size: Any | None = ...,
        metric_kwargs: Any | None = ...,
        strategy: Any | None = ...,
    ): ...
