from . import (
    EmpiricalCovariance as EmpiricalCovariance,
    empirical_covariance as empirical_covariance,
    log_likelihood as log_likelihood,
)
from ..exceptions import ConvergenceWarning as ConvergenceWarning
from ..linear_model import lars_path_gram as lars_path_gram
from ..model_selection import check_cv as check_cv, cross_val_score as cross_val_score
from ..utils._param_validation import (
    Interval as Interval,
    StrOptions as StrOptions,
    validate_params as validate_params,
)
from ..utils.parallel import Parallel as Parallel, delayed as delayed
from ..utils.validation import (
    check_random_state as check_random_state,
    check_scalar as check_scalar,
)
from typing import Any

def alpha_max(emp_cov): ...
def graphical_lasso(
    emp_cov,
    alpha,
    *,
    cov_init: Any | None = ...,
    mode: str = ...,
    tol: float = ...,
    enet_tol: float = ...,
    max_iter: int = ...,
    verbose: bool = ...,
    return_costs: bool = ...,
    eps=...,
    return_n_iter: bool = ...
): ...

class BaseGraphicalLasso(EmpiricalCovariance):
    tol: Any
    enet_tol: Any
    max_iter: Any
    mode: Any
    verbose: Any
    eps: Any
    def __init__(
        self,
        tol: float = ...,
        enet_tol: float = ...,
        max_iter: int = ...,
        mode: str = ...,
        verbose: bool = ...,
        eps=...,
        assume_centered: bool = ...,
    ) -> None: ...

class GraphicalLasso(BaseGraphicalLasso):
    alpha: Any
    covariance: Any
    def __init__(
        self,
        alpha: float = ...,
        *,
        mode: str = ...,
        covariance: Any | None = ...,
        tol: float = ...,
        enet_tol: float = ...,
        max_iter: int = ...,
        verbose: bool = ...,
        eps=...,
        assume_centered: bool = ...
    ) -> None: ...
    location_: Any
    def fit(self, X, y: Any | None = ...): ...

def graphical_lasso_path(
    X,
    alphas,
    cov_init: Any | None = ...,
    X_test: Any | None = ...,
    mode: str = ...,
    tol: float = ...,
    enet_tol: float = ...,
    max_iter: int = ...,
    verbose: bool = ...,
    eps=...,
): ...

class GraphicalLassoCV(BaseGraphicalLasso):
    alphas: Any
    n_refinements: Any
    cv: Any
    n_jobs: Any
    def __init__(
        self,
        *,
        alphas: int = ...,
        n_refinements: int = ...,
        cv: Any | None = ...,
        tol: float = ...,
        enet_tol: float = ...,
        max_iter: int = ...,
        mode: str = ...,
        n_jobs: Any | None = ...,
        verbose: bool = ...,
        eps=...,
        assume_centered: bool = ...
    ) -> None: ...
    location_: Any
    cv_results_: Any
    alpha_: Any
    def fit(self, X, y: Any | None = ...): ...
