from ..base import ClassifierMixin, RegressorMixin
from ._base import BaseEnsemble
from abc import ABCMeta, abstractmethod
from collections.abc import Generator
from typing import Any

class BaseWeightBoosting(BaseEnsemble, metaclass=ABCMeta):
    learning_rate: Any
    random_state: Any
    @abstractmethod
    def __init__(
        self,
        estimator: Any | None = ...,
        *,
        n_estimators: int = ...,
        estimator_params=...,
        learning_rate: float = ...,
        random_state: Any | None = ...,
        base_estimator: str = ...
    ): ...
    estimators_: Any
    estimator_weights_: Any
    estimator_errors_: Any
    def fit(self, X, y, sample_weight: Any | None = ...): ...
    def staged_score(
        self, X, y, sample_weight: Any | None = ...
    ) -> Generator[Any, None, None]: ...
    @property
    def feature_importances_(self): ...

class AdaBoostClassifier(ClassifierMixin, BaseWeightBoosting):
    algorithm: Any
    def __init__(
        self,
        estimator: Any | None = ...,
        *,
        n_estimators: int = ...,
        learning_rate: float = ...,
        algorithm: str = ...,
        random_state: Any | None = ...,
        base_estimator: str = ...
    ) -> None: ...
    def predict(self, X): ...
    def staged_predict(self, X) -> Generator[Any, None, None]: ...
    def decision_function(self, X): ...
    def staged_decision_function(self, X) -> Generator[Any, None, None]: ...
    def predict_proba(self, X): ...
    def staged_predict_proba(self, X) -> Generator[Any, None, None]: ...
    def predict_log_proba(self, X): ...

class AdaBoostRegressor(RegressorMixin, BaseWeightBoosting):
    loss: Any
    random_state: Any
    def __init__(
        self,
        estimator: Any | None = ...,
        *,
        n_estimators: int = ...,
        learning_rate: float = ...,
        loss: str = ...,
        random_state: Any | None = ...,
        base_estimator: str = ...
    ) -> None: ...
    def predict(self, X): ...
    def staged_predict(self, X) -> Generator[Any, None, None]: ...
