from .._config import config_context as config_context
from ..base import (
    BaseEstimator as BaseEstimator,
    ClassNamePrefixFeaturesOutMixin as ClassNamePrefixFeaturesOutMixin,
    TransformerMixin as TransformerMixin,
)
from ..exceptions import ConvergenceWarning as ConvergenceWarning
from ..utils import (
    check_array as check_array,
    check_random_state as check_random_state,
    gen_batches as gen_batches,
    metadata_routing as metadata_routing,
)
from ..utils._param_validation import (
    Interval as Interval,
    StrOptions as StrOptions,
    validate_params as validate_params,
)
from ..utils.extmath import (
    randomized_svd as randomized_svd,
    safe_sparse_dot as safe_sparse_dot,
    squared_norm as squared_norm,
)
from ..utils.validation import (
    check_is_fitted as check_is_fitted,
    check_non_negative as check_non_negative,
)
from abc import ABC
from typing import Any

EPSILON: Any

def norm(x): ...
def trace_dot(X, Y): ...
def non_negative_factorization(
    X,
    W: Any | None = ...,
    H: Any | None = ...,
    n_components: Any | None = ...,
    *,
    init: Any | None = ...,
    update_H: bool = ...,
    solver: str = ...,
    beta_loss: str = ...,
    tol: float = ...,
    max_iter: int = ...,
    alpha_W: float = ...,
    alpha_H: str = ...,
    l1_ratio: float = ...,
    random_state: Any | None = ...,
    verbose: int = ...,
    shuffle: bool = ...
): ...

class _BaseNMF(ClassNamePrefixFeaturesOutMixin, TransformerMixin, BaseEstimator, ABC):
    n_components: Any
    init: Any
    beta_loss: Any
    tol: Any
    max_iter: Any
    random_state: Any
    alpha_W: Any
    alpha_H: Any
    l1_ratio: Any
    verbose: Any
    def __init__(
        self,
        n_components: Any | None = ...,
        *,
        init: Any | None = ...,
        beta_loss: str = ...,
        tol: float = ...,
        max_iter: int = ...,
        random_state: Any | None = ...,
        alpha_W: float = ...,
        alpha_H: str = ...,
        l1_ratio: float = ...,
        verbose: int = ...
    ) -> None: ...
    def fit(self, X, y: Any | None = ..., **params): ...
    def inverse_transform(self, Xt: Any | None = ..., W: Any | None = ...): ...

class NMF(_BaseNMF):
    solver: Any
    shuffle: Any
    def __init__(
        self,
        n_components: Any | None = ...,
        *,
        init: Any | None = ...,
        solver: str = ...,
        beta_loss: str = ...,
        tol: float = ...,
        max_iter: int = ...,
        random_state: Any | None = ...,
        alpha_W: float = ...,
        alpha_H: str = ...,
        l1_ratio: float = ...,
        verbose: int = ...,
        shuffle: bool = ...
    ) -> None: ...
    reconstruction_err_: Any
    n_components_: Any
    components_: Any
    n_iter_: Any
    def fit_transform(
        self, X, y: Any | None = ..., W: Any | None = ..., H: Any | None = ...
    ): ...
    def transform(self, X): ...

class MiniBatchNMF(_BaseNMF):
    max_no_improvement: Any
    batch_size: Any
    forget_factor: Any
    fresh_restarts: Any
    fresh_restarts_max_iter: Any
    transform_max_iter: Any
    def __init__(
        self,
        n_components: Any | None = ...,
        *,
        init: Any | None = ...,
        batch_size: int = ...,
        beta_loss: str = ...,
        tol: float = ...,
        max_no_improvement: int = ...,
        max_iter: int = ...,
        alpha_W: float = ...,
        alpha_H: str = ...,
        l1_ratio: float = ...,
        forget_factor: float = ...,
        fresh_restarts: bool = ...,
        fresh_restarts_max_iter: int = ...,
        transform_max_iter: Any | None = ...,
        random_state: Any | None = ...,
        verbose: int = ...
    ) -> None: ...
    reconstruction_err_: Any
    n_components_: Any
    components_: Any
    n_iter_: Any
    n_steps_: Any
    def fit_transform(
        self, X, y: Any | None = ..., W: Any | None = ..., H: Any | None = ...
    ): ...
    def transform(self, X): ...
    def partial_fit(
        self, X, y: Any | None = ..., W: Any | None = ..., H: Any | None = ...
    ): ...
