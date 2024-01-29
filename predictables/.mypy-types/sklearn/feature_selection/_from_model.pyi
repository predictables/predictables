from ..base import (
    BaseEstimator as BaseEstimator,
    MetaEstimatorMixin as MetaEstimatorMixin,
    clone as clone,
)
from ..exceptions import NotFittedError as NotFittedError
from ..utils._param_validation import (
    HasMethods as HasMethods,
    Interval as Interval,
    Options as Options,
)
from ..utils.metaestimators import available_if as available_if
from ..utils.validation import (
    check_is_fitted as check_is_fitted,
    check_scalar as check_scalar,
)
from ._base import SelectorMixin as SelectorMixin
from typing import Any

class SelectFromModel(MetaEstimatorMixin, SelectorMixin, BaseEstimator):
    estimator: Any
    threshold: Any
    prefit: Any
    importance_getter: Any
    norm_order: Any
    max_features: Any
    def __init__(
        self,
        estimator,
        *,
        threshold: Any | None = ...,
        prefit: bool = ...,
        norm_order: int = ...,
        max_features: Any | None = ...,
        importance_getter: str = ...
    ) -> None: ...
    estimator_: Any
    feature_names_in_: Any
    def fit(self, X, y: Any | None = ..., **fit_params): ...
    @property
    def threshold_(self): ...
    def partial_fit(self, X, y: Any | None = ..., **fit_params): ...
    @property
    def n_features_in_(self): ...
