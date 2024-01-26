from ..._loss.loss import HalfSquaredError as HalfSquaredError
from ...exceptions import ConvergenceWarning as ConvergenceWarning
from .._linear_loss import LinearModelLoss as LinearModelLoss
from abc import ABC, abstractmethod
from typing import Any

class NewtonSolver(ABC):
    coef: Any
    linear_loss: Any
    l2_reg_strength: Any
    tol: Any
    max_iter: Any
    n_threads: Any
    verbose: Any
    def __init__(
        self,
        *,
        coef,
        linear_loss=...,
        l2_reg_strength: float = ...,
        tol: float = ...,
        max_iter: int = ...,
        n_threads: int = ...,
        verbose: int = ...
    ) -> None: ...
    loss_value: Any
    def setup(self, X, y, sample_weight) -> None: ...
    @abstractmethod
    def update_gradient_hessian(self, X, y, sample_weight): ...
    @abstractmethod
    def inner_solve(self, X, y, sample_weight): ...
    n_iter_: Any
    converged: Any
    def fallback_lbfgs_solve(self, X, y, sample_weight) -> None: ...
    coef_old: Any
    loss_value_old: Any
    gradient_old: Any
    use_fallback_lbfgs_solve: bool
    raw_prediction: Any
    def line_search(self, X, y, sample_weight) -> None: ...
    def check_convergence(self, X, y, sample_weight) -> None: ...
    def finalize(self, X, y, sample_weight) -> None: ...
    iteration: int
    def solve(self, X, y, sample_weight): ...

class NewtonCholeskySolver(NewtonSolver):
    gradient: Any
    hessian: Any
    def setup(self, X, y, sample_weight) -> None: ...
    def update_gradient_hessian(self, X, y, sample_weight) -> None: ...
    use_fallback_lbfgs_solve: bool
    coef_newton: Any
    gradient_times_newton: Any
    def inner_solve(self, X, y, sample_weight) -> None: ...
