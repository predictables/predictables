from ..base import ClassifierMixin, RegressorMixin
from ._base import BaseEnsemble
from abc import ABCMeta, abstractmethod
from typing import Any

class BaseBagging(BaseEnsemble, metaclass=ABCMeta):
    max_samples: Any
    max_features: Any
    bootstrap: Any
    bootstrap_features: Any
    oob_score: Any
    warm_start: Any
    n_jobs: Any
    random_state: Any
    verbose: Any
    @abstractmethod
    def __init__(
        self,
        estimator: Any | None = ...,
        n_estimators: int = ...,
        *,
        max_samples: float = ...,
        max_features: float = ...,
        bootstrap: bool = ...,
        bootstrap_features: bool = ...,
        oob_score: bool = ...,
        warm_start: bool = ...,
        n_jobs: Any | None = ...,
        random_state: Any | None = ...,
        verbose: int = ...,
        base_estimator: str = ...
    ): ...
    def fit(self, X, y, sample_weight: Any | None = ...): ...
    @property
    def estimators_samples_(self): ...

class BaggingClassifier(ClassifierMixin, BaseBagging):
    def __init__(
        self,
        estimator: Any | None = ...,
        n_estimators: int = ...,
        *,
        max_samples: float = ...,
        max_features: float = ...,
        bootstrap: bool = ...,
        bootstrap_features: bool = ...,
        oob_score: bool = ...,
        warm_start: bool = ...,
        n_jobs: Any | None = ...,
        random_state: Any | None = ...,
        verbose: int = ...,
        base_estimator: str = ...
    ) -> None: ...
    def predict(self, X): ...
    def predict_proba(self, X): ...
    def predict_log_proba(self, X): ...
    def decision_function(self, X): ...

class BaggingRegressor(RegressorMixin, BaseBagging):
    def __init__(
        self,
        estimator: Any | None = ...,
        n_estimators: int = ...,
        *,
        max_samples: float = ...,
        max_features: float = ...,
        bootstrap: bool = ...,
        bootstrap_features: bool = ...,
        oob_score: bool = ...,
        warm_start: bool = ...,
        n_jobs: Any | None = ...,
        random_state: Any | None = ...,
        verbose: int = ...,
        base_estimator: str = ...
    ) -> None: ...
    def predict(self, X): ...
