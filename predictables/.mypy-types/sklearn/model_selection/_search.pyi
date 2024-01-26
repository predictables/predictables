from ..base import BaseEstimator, MetaEstimatorMixin
from abc import ABCMeta, abstractmethod
from typing import Any

class ParameterGrid:
    param_grid: Any
    def __init__(self, param_grid) -> None: ...
    def __iter__(self): ...
    def __len__(self): ...
    def __getitem__(self, ind): ...

class ParameterSampler:
    n_iter: Any
    random_state: Any
    param_distributions: Any
    def __init__(
        self, param_distributions, n_iter, *, random_state: Any | None = ...
    ) -> None: ...
    def __iter__(self): ...
    def __len__(self): ...

class BaseSearchCV(MetaEstimatorMixin, BaseEstimator, metaclass=ABCMeta):
    scoring: Any
    estimator: Any
    n_jobs: Any
    refit: Any
    cv: Any
    verbose: Any
    pre_dispatch: Any
    error_score: Any
    return_train_score: Any
    @abstractmethod
    def __init__(
        self,
        estimator,
        *,
        scoring: Any | None = ...,
        n_jobs: Any | None = ...,
        refit: bool = ...,
        cv: Any | None = ...,
        verbose: int = ...,
        pre_dispatch: str = ...,
        error_score=...,
        return_train_score: bool = ...
    ): ...
    def score(self, X, y: Any | None = ...): ...
    def score_samples(self, X): ...
    def predict(self, X): ...
    def predict_proba(self, X): ...
    def predict_log_proba(self, X): ...
    def decision_function(self, X): ...
    def transform(self, X): ...
    def inverse_transform(self, Xt): ...
    @property
    def n_features_in_(self): ...
    @property
    def classes_(self): ...
    multimetric_: Any
    best_index_: Any
    best_score_: Any
    best_params_: Any
    best_estimator_: Any
    refit_time_: Any
    feature_names_in_: Any
    scorer_: Any
    cv_results_: Any
    n_splits_: Any
    def fit(
        self, X, y: Any | None = ..., *, groups: Any | None = ..., **fit_params
    ): ...

class GridSearchCV(BaseSearchCV):
    param_grid: Any
    def __init__(
        self,
        estimator,
        param_grid,
        *,
        scoring: Any | None = ...,
        n_jobs: Any | None = ...,
        refit: bool = ...,
        cv: Any | None = ...,
        verbose: int = ...,
        pre_dispatch: str = ...,
        error_score=...,
        return_train_score: bool = ...
    ) -> None: ...

class RandomizedSearchCV(BaseSearchCV):
    param_distributions: Any
    n_iter: Any
    random_state: Any
    def __init__(
        self,
        estimator,
        param_distributions,
        *,
        n_iter: int = ...,
        scoring: Any | None = ...,
        n_jobs: Any | None = ...,
        refit: bool = ...,
        cv: Any | None = ...,
        verbose: int = ...,
        pre_dispatch: str = ...,
        random_state: Any | None = ...,
        error_score=...,
        return_train_score: bool = ...
    ) -> None: ...
