from ..base import (
    ClassNamePrefixFeaturesOutMixin as ClassNamePrefixFeaturesOutMixin,
    TransformerMixin as TransformerMixin,
)
from ..utils._param_validation import StrOptions as StrOptions
from ..utils.validation import check_is_fitted as check_is_fitted
from ._base import (
    KNeighborsMixin as KNeighborsMixin,
    NeighborsBase as NeighborsBase,
    RadiusNeighborsMixin as RadiusNeighborsMixin,
)
from ._unsupervised import NearestNeighbors as NearestNeighbors
from typing import Any

def kneighbors_graph(
    X,
    n_neighbors,
    *,
    mode: str = ...,
    metric: str = ...,
    p: int = ...,
    metric_params: Any | None = ...,
    include_self: bool = ...,
    n_jobs: Any | None = ...
): ...
def radius_neighbors_graph(
    X,
    radius,
    *,
    mode: str = ...,
    metric: str = ...,
    p: int = ...,
    metric_params: Any | None = ...,
    include_self: bool = ...,
    n_jobs: Any | None = ...
): ...

class KNeighborsTransformer(
    ClassNamePrefixFeaturesOutMixin, KNeighborsMixin, TransformerMixin, NeighborsBase
):
    mode: Any
    def __init__(
        self,
        *,
        mode: str = ...,
        n_neighbors: int = ...,
        algorithm: str = ...,
        leaf_size: int = ...,
        metric: str = ...,
        p: int = ...,
        metric_params: Any | None = ...,
        n_jobs: Any | None = ...
    ) -> None: ...
    def fit(self, X, y: Any | None = ...): ...
    def transform(self, X): ...
    def fit_transform(self, X, y: Any | None = ...): ...

class RadiusNeighborsTransformer(
    ClassNamePrefixFeaturesOutMixin,
    RadiusNeighborsMixin,
    TransformerMixin,
    NeighborsBase,
):
    mode: Any
    def __init__(
        self,
        *,
        mode: str = ...,
        radius: float = ...,
        algorithm: str = ...,
        leaf_size: int = ...,
        metric: str = ...,
        p: int = ...,
        metric_params: Any | None = ...,
        n_jobs: Any | None = ...
    ) -> None: ...
    def fit(self, X, y: Any | None = ...): ...
    def transform(self, X): ...
    def fit_transform(self, X, y: Any | None = ...): ...
