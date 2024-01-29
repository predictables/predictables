from ._base import (
    KNeighborsMixin as KNeighborsMixin,
    NeighborsBase as NeighborsBase,
    RadiusNeighborsMixin as RadiusNeighborsMixin,
)
from typing import Any

class NearestNeighbors(KNeighborsMixin, RadiusNeighborsMixin, NeighborsBase):
    def __init__(
        self,
        *,
        n_neighbors: int = ...,
        radius: float = ...,
        algorithm: str = ...,
        leaf_size: int = ...,
        metric: str = ...,
        p: int = ...,
        metric_params: Any | None = ...,
        n_jobs: Any | None = ...
    ) -> None: ...
    def fit(self, X, y: Any | None = ...): ...
