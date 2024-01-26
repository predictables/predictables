from .base import BaseEstimator, ClassNamePrefixFeaturesOutMixin, TransformerMixin
from abc import ABCMeta, abstractmethod
from typing import Any

def johnson_lindenstrauss_min_dim(n_samples, *, eps: float = ...): ...

class BaseRandomProjection(
    TransformerMixin, BaseEstimator, ClassNamePrefixFeaturesOutMixin, metaclass=ABCMeta
):
    n_components: Any
    eps: Any
    compute_inverse_components: Any
    random_state: Any
    @abstractmethod
    def __init__(
        self,
        n_components: str = ...,
        *,
        eps: float = ...,
        compute_inverse_components: bool = ...,
        random_state: Any | None = ...
    ): ...
    n_components_: Any
    components_: Any
    inverse_components_: Any
    def fit(self, X, y: Any | None = ...): ...
    def inverse_transform(self, X): ...

class GaussianRandomProjection(BaseRandomProjection):
    def __init__(
        self,
        n_components: str = ...,
        *,
        eps: float = ...,
        compute_inverse_components: bool = ...,
        random_state: Any | None = ...
    ) -> None: ...
    def transform(self, X): ...

class SparseRandomProjection(BaseRandomProjection):
    dense_output: Any
    density: Any
    def __init__(
        self,
        n_components: str = ...,
        *,
        density: str = ...,
        eps: float = ...,
        dense_output: bool = ...,
        compute_inverse_components: bool = ...,
        random_state: Any | None = ...
    ) -> None: ...
    def transform(self, X): ...
