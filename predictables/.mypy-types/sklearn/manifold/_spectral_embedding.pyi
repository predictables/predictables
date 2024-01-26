from ..base import BaseEstimator as BaseEstimator
from ..metrics.pairwise import rbf_kernel as rbf_kernel
from ..neighbors import (
    NearestNeighbors as NearestNeighbors,
    kneighbors_graph as kneighbors_graph,
)
from ..utils import (
    check_array as check_array,
    check_random_state as check_random_state,
    check_symmetric as check_symmetric,
)
from ..utils._param_validation import Interval as Interval, StrOptions as StrOptions
from typing import Any

def spectral_embedding(
    adjacency,
    *,
    n_components: int = ...,
    eigen_solver: Any | None = ...,
    random_state: Any | None = ...,
    eigen_tol: str = ...,
    norm_laplacian: bool = ...,
    drop_first: bool = ...
): ...

class SpectralEmbedding(BaseEstimator):
    n_components: Any
    affinity: Any
    gamma: Any
    random_state: Any
    eigen_solver: Any
    eigen_tol: Any
    n_neighbors: Any
    n_jobs: Any
    def __init__(
        self,
        n_components: int = ...,
        *,
        affinity: str = ...,
        gamma: Any | None = ...,
        random_state: Any | None = ...,
        eigen_solver: Any | None = ...,
        eigen_tol: str = ...,
        n_neighbors: Any | None = ...,
        n_jobs: Any | None = ...
    ) -> None: ...
    embedding_: Any
    def fit(self, X, y: Any | None = ...): ...
    def fit_transform(self, X, y: Any | None = ...): ...
