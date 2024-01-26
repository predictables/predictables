from ..utils.extmath import squared_norm as squared_norm
from typing import Any

class LinearModelLoss:
    base_loss: Any
    fit_intercept: Any
    def __init__(self, base_loss, fit_intercept) -> None: ...
    def init_zero_coef(self, X, dtype: Any | None = ...): ...
    def weight_intercept(self, coef): ...
    def weight_intercept_raw(self, coef, X): ...
    def l2_penalty(self, weights, l2_reg_strength): ...
    def loss(
        self,
        coef,
        X,
        y,
        sample_weight: Any | None = ...,
        l2_reg_strength: float = ...,
        n_threads: int = ...,
        raw_prediction: Any | None = ...,
    ): ...
    def loss_gradient(
        self,
        coef,
        X,
        y,
        sample_weight: Any | None = ...,
        l2_reg_strength: float = ...,
        n_threads: int = ...,
        raw_prediction: Any | None = ...,
    ): ...
    def gradient(
        self,
        coef,
        X,
        y,
        sample_weight: Any | None = ...,
        l2_reg_strength: float = ...,
        n_threads: int = ...,
        raw_prediction: Any | None = ...,
    ): ...
    def gradient_hessian(
        self,
        coef,
        X,
        y,
        sample_weight: Any | None = ...,
        l2_reg_strength: float = ...,
        n_threads: int = ...,
        gradient_out: Any | None = ...,
        hessian_out: Any | None = ...,
        raw_prediction: Any | None = ...,
    ): ...
    def gradient_hessian_product(
        self,
        coef,
        X,
        y,
        sample_weight: Any | None = ...,
        l2_reg_strength: float = ...,
        n_threads: int = ...,
    ): ...
