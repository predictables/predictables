from .base import (
    BaseEstimator as BaseEstimator,
    ClassNamePrefixFeaturesOutMixin as ClassNamePrefixFeaturesOutMixin,
    TransformerMixin as TransformerMixin,
)
from .metrics.pairwise import (
    KERNEL_PARAMS as KERNEL_PARAMS,
    PAIRWISE_KERNEL_FUNCTIONS as PAIRWISE_KERNEL_FUNCTIONS,
    pairwise_kernels as pairwise_kernels,
)
from .utils import check_random_state as check_random_state, deprecated as deprecated
from .utils._param_validation import Interval as Interval, StrOptions as StrOptions
from .utils.extmath import safe_sparse_dot as safe_sparse_dot
from .utils.validation import (
    check_is_fitted as check_is_fitted,
    check_non_negative as check_non_negative,
)
from typing import Any

class PolynomialCountSketch(
    ClassNamePrefixFeaturesOutMixin, TransformerMixin, BaseEstimator
):
    gamma: Any
    degree: Any
    coef0: Any
    n_components: Any
    random_state: Any
    def __init__(
        self,
        *,
        gamma: float = ...,
        degree: int = ...,
        coef0: int = ...,
        n_components: int = ...,
        random_state: Any | None = ...
    ) -> None: ...
    indexHash_: Any
    bitHash_: Any
    def fit(self, X, y: Any | None = ...): ...
    def transform(self, X): ...

class RBFSampler(ClassNamePrefixFeaturesOutMixin, TransformerMixin, BaseEstimator):
    gamma: Any
    n_components: Any
    random_state: Any
    def __init__(
        self,
        *,
        gamma: float = ...,
        n_components: int = ...,
        random_state: Any | None = ...
    ) -> None: ...
    random_weights_: Any
    random_offset_: Any
    def fit(self, X, y: Any | None = ...): ...
    def transform(self, X): ...

class SkewedChi2Sampler(
    ClassNamePrefixFeaturesOutMixin, TransformerMixin, BaseEstimator
):
    skewedness: Any
    n_components: Any
    random_state: Any
    def __init__(
        self,
        *,
        skewedness: float = ...,
        n_components: int = ...,
        random_state: Any | None = ...
    ) -> None: ...
    random_weights_: Any
    random_offset_: Any
    def fit(self, X, y: Any | None = ...): ...
    def transform(self, X): ...

class AdditiveChi2Sampler(TransformerMixin, BaseEstimator):
    sample_steps: Any
    sample_interval: Any
    def __init__(
        self, *, sample_steps: int = ..., sample_interval: Any | None = ...
    ) -> None: ...
    def fit(self, X, y: Any | None = ...): ...
    @property
    def sample_interval_(self): ...
    def transform(self, X): ...
    def get_feature_names_out(self, input_features: Any | None = ...): ...

class Nystroem(ClassNamePrefixFeaturesOutMixin, TransformerMixin, BaseEstimator):
    kernel: Any
    gamma: Any
    coef0: Any
    degree: Any
    kernel_params: Any
    n_components: Any
    random_state: Any
    n_jobs: Any
    def __init__(
        self,
        kernel: str = ...,
        *,
        gamma: Any | None = ...,
        coef0: Any | None = ...,
        degree: Any | None = ...,
        kernel_params: Any | None = ...,
        n_components: int = ...,
        random_state: Any | None = ...,
        n_jobs: Any | None = ...
    ) -> None: ...
    normalization_: Any
    components_: Any
    component_indices_: Any
    def fit(self, X, y: Any | None = ...): ...
    def transform(self, X): ...
