from ..utils._param_validation import Interval as Interval, StrOptions as StrOptions
from ._stochastic_gradient import (
    BaseSGDClassifier as BaseSGDClassifier,
    BaseSGDRegressor as BaseSGDRegressor,
    DEFAULT_EPSILON as DEFAULT_EPSILON,
)
from typing import Any

class PassiveAggressiveClassifier(BaseSGDClassifier):
    C: Any
    loss: Any
    def __init__(
        self,
        *,
        C: float = ...,
        fit_intercept: bool = ...,
        max_iter: int = ...,
        tol: float = ...,
        early_stopping: bool = ...,
        validation_fraction: float = ...,
        n_iter_no_change: int = ...,
        shuffle: bool = ...,
        verbose: int = ...,
        loss: str = ...,
        n_jobs: Any | None = ...,
        random_state: Any | None = ...,
        warm_start: bool = ...,
        class_weight: Any | None = ...,
        average: bool = ...
    ) -> None: ...
    def partial_fit(self, X, y, classes: Any | None = ...): ...
    def fit(
        self, X, y, coef_init: Any | None = ..., intercept_init: Any | None = ...
    ): ...

class PassiveAggressiveRegressor(BaseSGDRegressor):
    C: Any
    loss: Any
    def __init__(
        self,
        *,
        C: float = ...,
        fit_intercept: bool = ...,
        max_iter: int = ...,
        tol: float = ...,
        early_stopping: bool = ...,
        validation_fraction: float = ...,
        n_iter_no_change: int = ...,
        shuffle: bool = ...,
        verbose: int = ...,
        loss: str = ...,
        epsilon=...,
        random_state: Any | None = ...,
        warm_start: bool = ...,
        average: bool = ...
    ) -> None: ...
    def partial_fit(self, X, y): ...
    def fit(
        self, X, y, coef_init: Any | None = ..., intercept_init: Any | None = ...
    ): ...
