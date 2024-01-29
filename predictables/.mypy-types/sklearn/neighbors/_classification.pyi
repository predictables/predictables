from ..base import ClassifierMixin as ClassifierMixin
from ..metrics._pairwise_distances_reduction import ArgKminClassMode as ArgKminClassMode
from ..utils._param_validation import StrOptions as StrOptions
from ..utils.extmath import weighted_mode as weighted_mode
from ..utils.validation import check_is_fitted as check_is_fitted
from ._base import (
    KNeighborsMixin as KNeighborsMixin,
    NeighborsBase as NeighborsBase,
    RadiusNeighborsMixin as RadiusNeighborsMixin,
)
from typing import Any

class KNeighborsClassifier(KNeighborsMixin, ClassifierMixin, NeighborsBase):
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
    def predict_proba(self, X): ...

class RadiusNeighborsClassifier(RadiusNeighborsMixin, ClassifierMixin, NeighborsBase):
    weights: Any
    outlier_label: Any
    def __init__(
        self,
        radius: float = ...,
        *,
        weights: str = ...,
        algorithm: str = ...,
        leaf_size: int = ...,
        p: int = ...,
        metric: str = ...,
        outlier_label: Any | None = ...,
        metric_params: Any | None = ...,
        n_jobs: Any | None = ...
    ) -> None: ...
    outlier_label_: Any
    def fit(self, X, y): ...
    def predict(self, X): ...
    def predict_proba(self, X): ...
