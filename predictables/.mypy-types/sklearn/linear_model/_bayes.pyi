from ..base import RegressorMixin as RegressorMixin
from ..utils._param_validation import (
    Hidden as Hidden,
    Interval as Interval,
    StrOptions as StrOptions,
)
from ..utils.extmath import fast_logdet as fast_logdet
from ._base import LinearModel as LinearModel
from typing import Any

class BayesianRidge(RegressorMixin, LinearModel):
    max_iter: Any
    tol: Any
    alpha_1: Any
    alpha_2: Any
    lambda_1: Any
    lambda_2: Any
    alpha_init: Any
    lambda_init: Any
    compute_score: Any
    fit_intercept: Any
    copy_X: Any
    verbose: Any
    n_iter: Any
    def __init__(
        self,
        *,
        max_iter: Any | None = ...,
        tol: float = ...,
        alpha_1: float = ...,
        alpha_2: float = ...,
        lambda_1: float = ...,
        lambda_2: float = ...,
        alpha_init: Any | None = ...,
        lambda_init: Any | None = ...,
        compute_score: bool = ...,
        fit_intercept: bool = ...,
        copy_X: bool = ...,
        verbose: bool = ...,
        n_iter: str = ...
    ) -> None: ...
    X_offset_: Any
    X_scale_: Any
    scores_: Any
    n_iter_: Any
    alpha_: Any
    lambda_: Any
    sigma_: Any
    def fit(self, X, y, sample_weight: Any | None = ...): ...
    def predict(self, X, return_std: bool = ...): ...

class ARDRegression(RegressorMixin, LinearModel):
    max_iter: Any
    tol: Any
    fit_intercept: Any
    alpha_1: Any
    alpha_2: Any
    lambda_1: Any
    lambda_2: Any
    compute_score: Any
    threshold_lambda: Any
    copy_X: Any
    verbose: Any
    n_iter: Any
    def __init__(
        self,
        *,
        max_iter: Any | None = ...,
        tol: float = ...,
        alpha_1: float = ...,
        alpha_2: float = ...,
        lambda_1: float = ...,
        lambda_2: float = ...,
        compute_score: bool = ...,
        threshold_lambda: float = ...,
        fit_intercept: bool = ...,
        copy_X: bool = ...,
        verbose: bool = ...,
        n_iter: str = ...
    ) -> None: ...
    X_offset_: Any
    X_scale_: Any
    scores_: Any
    n_iter_: Any
    coef_: Any
    alpha_: Any
    sigma_: Any
    lambda_: Any
    def fit(self, X, y): ...
    def predict(self, X, return_std: bool = ...): ...
