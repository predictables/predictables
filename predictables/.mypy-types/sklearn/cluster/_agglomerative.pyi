from ..base import (
    BaseEstimator as BaseEstimator,
    ClassNamePrefixFeaturesOutMixin as ClassNamePrefixFeaturesOutMixin,
    ClusterMixin as ClusterMixin,
)
from ..metrics import DistanceMetric as DistanceMetric
from ..metrics._dist_metrics import METRIC_MAPPING64 as METRIC_MAPPING64
from ..metrics.pairwise import paired_distances as paired_distances
from ..utils import check_array as check_array
from ..utils._fast_dict import IntFloatDict as IntFloatDict
from ..utils._param_validation import (
    HasMethods as HasMethods,
    Hidden as Hidden,
    Interval as Interval,
    StrOptions as StrOptions,
    validate_params as validate_params,
)
from ..utils.validation import check_memory as check_memory
from ._feature_agglomeration import AgglomerationTransform as AgglomerationTransform
from typing import Any

def ward_tree(
    X,
    *,
    connectivity: Any | None = ...,
    n_clusters: Any | None = ...,
    return_distance: bool = ...
): ...
def linkage_tree(
    X,
    connectivity: Any | None = ...,
    n_clusters: Any | None = ...,
    linkage: str = ...,
    affinity: str = ...,
    return_distance: bool = ...,
): ...

class AgglomerativeClustering(ClusterMixin, BaseEstimator):
    n_clusters: Any
    distance_threshold: Any
    memory: Any
    connectivity: Any
    compute_full_tree: Any
    linkage: Any
    affinity: Any
    metric: Any
    compute_distances: Any
    def __init__(
        self,
        n_clusters: int = ...,
        *,
        affinity: str = ...,
        metric: Any | None = ...,
        memory: Any | None = ...,
        connectivity: Any | None = ...,
        compute_full_tree: str = ...,
        linkage: str = ...,
        distance_threshold: Any | None = ...,
        compute_distances: bool = ...
    ) -> None: ...
    def fit(self, X, y: Any | None = ...): ...
    def fit_predict(self, X, y: Any | None = ...): ...

class FeatureAgglomeration(
    ClassNamePrefixFeaturesOutMixin, AgglomerativeClustering, AgglomerationTransform
):
    pooling_func: Any
    def __init__(
        self,
        n_clusters: int = ...,
        *,
        affinity: str = ...,
        metric: Any | None = ...,
        memory: Any | None = ...,
        connectivity: Any | None = ...,
        compute_full_tree: str = ...,
        linkage: str = ...,
        pooling_func=...,
        distance_threshold: Any | None = ...,
        compute_distances: bool = ...
    ) -> None: ...
    def fit(self, X, y: Any | None = ...): ...
    @property
    def fit_predict(self) -> None: ...
