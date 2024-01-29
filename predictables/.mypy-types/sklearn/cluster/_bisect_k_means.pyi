from ..utils._param_validation import StrOptions as StrOptions
from ..utils.extmath import row_norms as row_norms
from ..utils.validation import (
    check_is_fitted as check_is_fitted,
    check_random_state as check_random_state,
)
from ._kmeans import _BaseKMeans
from collections.abc import Generator
from typing import Any

class _BisectingTree:
    center: Any
    indices: Any
    score: Any
    left: Any
    right: Any
    def __init__(self, center, indices, score) -> None: ...
    def split(self, labels, centers, scores) -> None: ...
    def get_cluster_to_bisect(self): ...
    def iter_leaves(self) -> Generator[Any, None, None]: ...

class BisectingKMeans(_BaseKMeans):
    copy_x: Any
    algorithm: Any
    bisecting_strategy: Any
    def __init__(
        self,
        n_clusters: int = ...,
        *,
        init: str = ...,
        n_init: int = ...,
        random_state: Any | None = ...,
        max_iter: int = ...,
        verbose: int = ...,
        tol: float = ...,
        copy_x: bool = ...,
        algorithm: str = ...,
        bisecting_strategy: str = ...
    ) -> None: ...
    labels_: Any
    cluster_centers_: Any
    inertia_: Any
    def fit(self, X, y: Any | None = ..., sample_weight: Any | None = ...): ...
    def predict(self, X): ...
