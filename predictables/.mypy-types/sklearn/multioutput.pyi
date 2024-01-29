from .base import BaseEstimator, ClassifierMixin, MetaEstimatorMixin, RegressorMixin
from abc import ABCMeta, abstractmethod
from typing import Any

class _MultiOutputEstimator(MetaEstimatorMixin, BaseEstimator, metaclass=ABCMeta):
    estimator: Any
    n_jobs: Any
    @abstractmethod
    def __init__(self, estimator, *, n_jobs: Any | None = ...): ...
    estimators_: Any
    n_features_in_: Any
    feature_names_in_: Any
    def partial_fit(
        self,
        X,
        y,
        classes: Any | None = ...,
        sample_weight: Any | None = ...,
        **partial_fit_params
    ): ...
    def fit(self, X, y, sample_weight: Any | None = ..., **fit_params): ...
    def predict(self, X): ...
    def get_metadata_routing(self): ...

class MultiOutputRegressor(RegressorMixin, _MultiOutputEstimator):
    def __init__(self, estimator, *, n_jobs: Any | None = ...) -> None: ...
    def partial_fit(
        self, X, y, sample_weight: Any | None = ..., **partial_fit_params
    ) -> None: ...

class MultiOutputClassifier(ClassifierMixin, _MultiOutputEstimator):
    def __init__(self, estimator, *, n_jobs: Any | None = ...) -> None: ...
    classes_: Any
    def fit(self, X, Y, sample_weight: Any | None = ..., **fit_params): ...
    def predict_proba(self, X): ...
    def score(self, X, y): ...

class _BaseChain(BaseEstimator, metaclass=ABCMeta):
    base_estimator: Any
    order: Any
    cv: Any
    random_state: Any
    verbose: Any
    def __init__(
        self,
        base_estimator,
        *,
        order: Any | None = ...,
        cv: Any | None = ...,
        random_state: Any | None = ...,
        verbose: bool = ...
    ) -> None: ...
    order_: Any
    estimators_: Any
    @abstractmethod
    def fit(self, X, Y, **fit_params): ...
    def predict(self, X): ...

class ClassifierChain(MetaEstimatorMixin, ClassifierMixin, _BaseChain):
    classes_: Any
    def fit(self, X, Y, **fit_params): ...
    def predict_proba(self, X): ...
    def decision_function(self, X): ...
    def get_metadata_routing(self): ...

class RegressorChain(MetaEstimatorMixin, RegressorMixin, _BaseChain):
    def fit(self, X, Y, **fit_params): ...
    def get_metadata_routing(self): ...
