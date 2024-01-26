from ..._loss.loss import (
    HalfGammaLoss as HalfGammaLoss,
    HalfPoissonLoss as HalfPoissonLoss,
    HalfSquaredError as HalfSquaredError,
    HalfTweedieLoss as HalfTweedieLoss,
    HalfTweedieLossIdentity as HalfTweedieLossIdentity,
)
from ...base import BaseEstimator as BaseEstimator, RegressorMixin as RegressorMixin
from ...utils import check_array as check_array
from ...utils._param_validation import (
    Hidden as Hidden,
    Interval as Interval,
    StrOptions as StrOptions,
)
from ...utils.validation import check_is_fitted as check_is_fitted
from .._linear_loss import LinearModelLoss as LinearModelLoss
from ._newton_solver import (
    NewtonCholeskySolver as NewtonCholeskySolver,
    NewtonSolver as NewtonSolver,
)
from typing import Any

class _GeneralizedLinearRegressor(RegressorMixin, BaseEstimator):
    alpha: Any
    fit_intercept: Any
    solver: Any
    max_iter: Any
    tol: Any
    warm_start: Any
    verbose: Any
    def __init__(
        self,
        *,
        alpha: float = ...,
        fit_intercept: bool = ...,
        solver: str = ...,
        max_iter: int = ...,
        tol: float = ...,
        warm_start: bool = ...,
        verbose: int = ...
    ) -> None: ...
    n_iter_: Any
    intercept_: Any
    coef_: Any
    def fit(self, X, y, sample_weight: Any | None = ...): ...
    def predict(self, X): ...
    def score(self, X, y, sample_weight: Any | None = ...): ...

class PoissonRegressor(_GeneralizedLinearRegressor):
    def __init__(
        self,
        *,
        alpha: float = ...,
        fit_intercept: bool = ...,
        solver: str = ...,
        max_iter: int = ...,
        tol: float = ...,
        warm_start: bool = ...,
        verbose: int = ...
    ) -> None: ...

class GammaRegressor(_GeneralizedLinearRegressor):
    def __init__(
        self,
        *,
        alpha: float = ...,
        fit_intercept: bool = ...,
        solver: str = ...,
        max_iter: int = ...,
        tol: float = ...,
        warm_start: bool = ...,
        verbose: int = ...
    ) -> None: ...

class TweedieRegressor(_GeneralizedLinearRegressor):
    link: Any
    power: Any
    def __init__(
        self,
        *,
        power: float = ...,
        alpha: float = ...,
        fit_intercept: bool = ...,
        link: str = ...,
        solver: str = ...,
        max_iter: int = ...,
        tol: float = ...,
        warm_start: bool = ...,
        verbose: int = ...
    ) -> None: ...
