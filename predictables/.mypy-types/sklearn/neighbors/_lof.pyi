from ..base import OutlierMixin
from ._base import KNeighborsMixin, NeighborsBase
from typing import Any

class LocalOutlierFactor(KNeighborsMixin, OutlierMixin, NeighborsBase):
    contamination: Any
    novelty: Any
    def __init__(
        self,
        n_neighbors: int = ...,
        *,
        algorithm: str = ...,
        leaf_size: int = ...,
        metric: str = ...,
        p: int = ...,
        metric_params: Any | None = ...,
        contamination: str = ...,
        novelty: bool = ...,
        n_jobs: Any | None = ...
    ) -> None: ...
    def fit_predict(self, X, y: Any | None = ...): ...
    n_neighbors_: Any
    negative_outlier_factor_: Any
    offset_: Any
    def fit(self, X, y: Any | None = ...): ...
    def predict(self, X: Any | None = ...): ...
    def decision_function(self, X): ...
    def score_samples(self, X): ...
