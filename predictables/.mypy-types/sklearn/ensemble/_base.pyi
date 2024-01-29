from ..base import (
    BaseEstimator as BaseEstimator,
    MetaEstimatorMixin as MetaEstimatorMixin,
    clone as clone,
    is_classifier as is_classifier,
    is_regressor as is_regressor,
)
from ..utils import (
    Bunch as Bunch,
    check_random_state as check_random_state,
    deprecated as deprecated,
)
from ..utils.metaestimators import _BaseComposition
from abc import ABCMeta, abstractmethod
from typing import Any

class BaseEnsemble(MetaEstimatorMixin, BaseEstimator, metaclass=ABCMeta):
    estimator: Any
    n_estimators: Any
    estimator_params: Any
    base_estimator: Any
    @abstractmethod
    def __init__(
        self,
        estimator: Any | None = ...,
        *,
        n_estimators: int = ...,
        estimator_params=...,
        base_estimator: str = ...
    ): ...
    @property
    def base_estimator_(self): ...
    def __len__(self): ...
    def __getitem__(self, index): ...
    def __iter__(self): ...

class _BaseHeterogeneousEnsemble(
    MetaEstimatorMixin, _BaseComposition, metaclass=ABCMeta
):
    @property
    def named_estimators(self): ...
    estimators: Any
    @abstractmethod
    def __init__(self, estimators): ...
    def set_params(self, **params): ...
    def get_params(self, deep: bool = ...): ...
