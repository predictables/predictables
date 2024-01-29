from ...base import BaseEstimator as BaseEstimator, ClusterMixin as ClusterMixin
from ...metrics import pairwise_distances as pairwise_distances
from ...metrics._dist_metrics import DistanceMetric as DistanceMetric
from ...neighbors import (
    BallTree as BallTree,
    KDTree as KDTree,
    NearestNeighbors as NearestNeighbors,
)
from ...utils._param_validation import Interval as Interval, StrOptions as StrOptions
from ._linkage import (
    MST_edge_dtype as MST_edge_dtype,
    make_single_linkage as make_single_linkage,
    mst_from_data_matrix as mst_from_data_matrix,
    mst_from_mutual_reachability as mst_from_mutual_reachability,
)
from ._reachability import mutual_reachability_graph as mutual_reachability_graph
from ._tree import (
    HIERARCHY_dtype as HIERARCHY_dtype,
    labelling_at_cut as labelling_at_cut,
    tree_to_labels as tree_to_labels,
)
from typing import Any

FAST_METRICS: Any

def remap_single_linkage_tree(tree, internal_to_raw, non_finite): ...

class HDBSCAN(ClusterMixin, BaseEstimator):
    min_cluster_size: Any
    min_samples: Any
    alpha: Any
    max_cluster_size: Any
    cluster_selection_epsilon: Any
    metric: Any
    metric_params: Any
    algorithm: Any
    leaf_size: Any
    n_jobs: Any
    cluster_selection_method: Any
    allow_single_cluster: Any
    store_centers: Any
    copy: Any
    def __init__(
        self,
        min_cluster_size: int = ...,
        min_samples: Any | None = ...,
        cluster_selection_epsilon: float = ...,
        max_cluster_size: Any | None = ...,
        metric: str = ...,
        metric_params: Any | None = ...,
        alpha: float = ...,
        algorithm: str = ...,
        leaf_size: int = ...,
        n_jobs: Any | None = ...,
        cluster_selection_method: str = ...,
        allow_single_cluster: bool = ...,
        store_centers: Any | None = ...,
        copy: bool = ...,
    ) -> None: ...
    labels_: Any
    probabilities_: Any
    def fit(self, X, y: Any | None = ...): ...
    def fit_predict(self, X, y: Any | None = ...): ...
    def dbscan_clustering(self, cut_distance, min_cluster_size: int = ...): ...
