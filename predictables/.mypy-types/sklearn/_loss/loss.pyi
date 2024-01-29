from ..utils import check_scalar as check_scalar
from ._loss import (
    CyAbsoluteError as CyAbsoluteError,
    CyExponentialLoss as CyExponentialLoss,
    CyHalfBinomialLoss as CyHalfBinomialLoss,
    CyHalfGammaLoss as CyHalfGammaLoss,
    CyHalfMultinomialLoss as CyHalfMultinomialLoss,
    CyHalfPoissonLoss as CyHalfPoissonLoss,
    CyHalfSquaredError as CyHalfSquaredError,
    CyHalfTweedieLoss as CyHalfTweedieLoss,
    CyHalfTweedieLossIdentity as CyHalfTweedieLossIdentity,
    CyHuberLoss as CyHuberLoss,
    CyPinballLoss as CyPinballLoss,
)
from .link import (
    HalfLogitLink as HalfLogitLink,
    IdentityLink as IdentityLink,
    Interval as Interval,
    LogLink as LogLink,
    LogitLink as LogitLink,
    MultinomialLogit as MultinomialLogit,
)
from typing import Any

class BaseLoss:
    need_update_leaves_values: bool
    differentiable: bool
    is_multiclass: bool
    closs: Any
    link: Any
    approx_hessian: bool
    constant_hessian: bool
    n_classes: Any
    interval_y_true: Any
    interval_y_pred: Any
    def __init__(self, closs, link, n_classes: Any | None = ...) -> None: ...
    def in_y_true_range(self, y): ...
    def in_y_pred_range(self, y): ...
    def loss(
        self,
        y_true,
        raw_prediction,
        sample_weight: Any | None = ...,
        loss_out: Any | None = ...,
        n_threads: int = ...,
    ): ...
    def loss_gradient(
        self,
        y_true,
        raw_prediction,
        sample_weight: Any | None = ...,
        loss_out: Any | None = ...,
        gradient_out: Any | None = ...,
        n_threads: int = ...,
    ): ...
    def gradient(
        self,
        y_true,
        raw_prediction,
        sample_weight: Any | None = ...,
        gradient_out: Any | None = ...,
        n_threads: int = ...,
    ): ...
    def gradient_hessian(
        self,
        y_true,
        raw_prediction,
        sample_weight: Any | None = ...,
        gradient_out: Any | None = ...,
        hessian_out: Any | None = ...,
        n_threads: int = ...,
    ): ...
    def __call__(
        self,
        y_true,
        raw_prediction,
        sample_weight: Any | None = ...,
        n_threads: int = ...,
    ): ...
    def fit_intercept_only(self, y_true, sample_weight: Any | None = ...): ...
    def constant_to_optimal_zero(self, y_true, sample_weight: Any | None = ...): ...
    def init_gradient_and_hessian(self, n_samples, dtype=..., order: str = ...): ...

class HalfSquaredError(BaseLoss):
    constant_hessian: Any
    def __init__(self, sample_weight: Any | None = ...) -> None: ...

class AbsoluteError(BaseLoss):
    differentiable: bool
    need_update_leaves_values: bool
    approx_hessian: bool
    constant_hessian: Any
    def __init__(self, sample_weight: Any | None = ...) -> None: ...
    def fit_intercept_only(self, y_true, sample_weight: Any | None = ...): ...

class PinballLoss(BaseLoss):
    differentiable: bool
    need_update_leaves_values: bool
    approx_hessian: bool
    constant_hessian: Any
    def __init__(
        self, sample_weight: Any | None = ..., quantile: float = ...
    ) -> None: ...
    def fit_intercept_only(self, y_true, sample_weight: Any | None = ...): ...

class HuberLoss(BaseLoss):
    differentiable: bool
    need_update_leaves_values: bool
    quantile: Any
    approx_hessian: bool
    constant_hessian: bool
    def __init__(
        self, sample_weight: Any | None = ..., quantile: float = ..., delta: float = ...
    ) -> None: ...
    def fit_intercept_only(self, y_true, sample_weight: Any | None = ...): ...

class HalfPoissonLoss(BaseLoss):
    interval_y_true: Any
    def __init__(self, sample_weight: Any | None = ...) -> None: ...
    def constant_to_optimal_zero(self, y_true, sample_weight: Any | None = ...): ...

class HalfGammaLoss(BaseLoss):
    interval_y_true: Any
    def __init__(self, sample_weight: Any | None = ...) -> None: ...
    def constant_to_optimal_zero(self, y_true, sample_weight: Any | None = ...): ...

class HalfTweedieLoss(BaseLoss):
    interval_y_true: Any
    def __init__(self, sample_weight: Any | None = ..., power: float = ...) -> None: ...
    def constant_to_optimal_zero(self, y_true, sample_weight: Any | None = ...): ...

class HalfTweedieLossIdentity(BaseLoss):
    interval_y_true: Any
    interval_y_pred: Any
    def __init__(self, sample_weight: Any | None = ..., power: float = ...) -> None: ...

class HalfBinomialLoss(BaseLoss):
    interval_y_true: Any
    def __init__(self, sample_weight: Any | None = ...) -> None: ...
    def constant_to_optimal_zero(self, y_true, sample_weight: Any | None = ...): ...
    def predict_proba(self, raw_prediction): ...

class HalfMultinomialLoss(BaseLoss):
    is_multiclass: bool
    interval_y_true: Any
    interval_y_pred: Any
    def __init__(
        self, sample_weight: Any | None = ..., n_classes: int = ...
    ) -> None: ...
    def in_y_true_range(self, y): ...
    def fit_intercept_only(self, y_true, sample_weight: Any | None = ...): ...
    def predict_proba(self, raw_prediction): ...
    def gradient_proba(
        self,
        y_true,
        raw_prediction,
        sample_weight: Any | None = ...,
        gradient_out: Any | None = ...,
        proba_out: Any | None = ...,
        n_threads: int = ...,
    ): ...

class ExponentialLoss(BaseLoss):
    interval_y_true: Any
    def __init__(self, sample_weight: Any | None = ...) -> None: ...
    def constant_to_optimal_zero(self, y_true, sample_weight: Any | None = ...): ...
    def predict_proba(self, raw_prediction): ...
