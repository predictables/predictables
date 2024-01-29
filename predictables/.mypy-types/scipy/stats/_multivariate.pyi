from typing import Any

class _PSD:
    eps: Any
    V: Any
    rank: Any
    U: Any
    log_pdet: Any
    def __init__(
        self,
        M,
        cond: Any | None = ...,
        rcond: Any | None = ...,
        lower: bool = ...,
        check_finite: bool = ...,
        allow_singular: bool = ...,
    ) -> None: ...
    @property
    def pinv(self): ...

class multi_rv_generic:
    def __init__(self, seed: Any | None = ...) -> None: ...
    @property
    def random_state(self): ...
    @random_state.setter
    def random_state(self, seed) -> None: ...

class multi_rv_frozen:
    @property
    def random_state(self): ...
    @random_state.setter
    def random_state(self, seed) -> None: ...

class multivariate_normal_gen(multi_rv_generic):
    __doc__: Any
    def __init__(self, seed: Any | None = ...) -> None: ...
    def __call__(
        self,
        mean: Any | None = ...,
        cov: int = ...,
        allow_singular: bool = ...,
        seed: Any | None = ...,
    ): ...
    def logpdf(
        self, x, mean: Any | None = ..., cov: int = ..., allow_singular: bool = ...
    ): ...
    def pdf(
        self, x, mean: Any | None = ..., cov: int = ..., allow_singular: bool = ...
    ): ...
    def logcdf(
        self,
        x,
        mean: Any | None = ...,
        cov: int = ...,
        allow_singular: bool = ...,
        maxpts: Any | None = ...,
        abseps: float = ...,
        releps: float = ...,
        *,
        lower_limit: Any | None = ...
    ): ...
    def cdf(
        self,
        x,
        mean: Any | None = ...,
        cov: int = ...,
        allow_singular: bool = ...,
        maxpts: Any | None = ...,
        abseps: float = ...,
        releps: float = ...,
        *,
        lower_limit: Any | None = ...
    ): ...
    def rvs(
        self,
        mean: Any | None = ...,
        cov: int = ...,
        size: int = ...,
        random_state: Any | None = ...,
    ): ...
    def entropy(self, mean: Any | None = ..., cov: int = ...): ...

multivariate_normal: Any

class multivariate_normal_frozen(multi_rv_frozen):
    allow_singular: Any
    maxpts: Any
    abseps: Any
    releps: Any
    def __init__(
        self,
        mean: Any | None = ...,
        cov: int = ...,
        allow_singular: bool = ...,
        seed: Any | None = ...,
        maxpts: Any | None = ...,
        abseps: float = ...,
        releps: float = ...,
    ) -> None: ...
    @property
    def cov(self): ...
    def logpdf(self, x): ...
    def pdf(self, x): ...
    def logcdf(self, x, *, lower_limit: Any | None = ...): ...
    def cdf(self, x, *, lower_limit: Any | None = ...): ...
    def rvs(self, size: int = ..., random_state: Any | None = ...): ...
    def entropy(self): ...

class matrix_normal_gen(multi_rv_generic):
    __doc__: Any
    def __init__(self, seed: Any | None = ...) -> None: ...
    def __call__(
        self,
        mean: Any | None = ...,
        rowcov: int = ...,
        colcov: int = ...,
        seed: Any | None = ...,
    ): ...
    def logpdf(
        self, X, mean: Any | None = ..., rowcov: int = ..., colcov: int = ...
    ): ...
    def pdf(self, X, mean: Any | None = ..., rowcov: int = ..., colcov: int = ...): ...
    def rvs(
        self,
        mean: Any | None = ...,
        rowcov: int = ...,
        colcov: int = ...,
        size: int = ...,
        random_state: Any | None = ...,
    ): ...
    def entropy(self, rowcov: int = ..., colcov: int = ...): ...

matrix_normal: Any

class matrix_normal_frozen(multi_rv_frozen):
    rowpsd: Any
    colpsd: Any
    def __init__(
        self,
        mean: Any | None = ...,
        rowcov: int = ...,
        colcov: int = ...,
        seed: Any | None = ...,
    ) -> None: ...
    def logpdf(self, X): ...
    def pdf(self, X): ...
    def rvs(self, size: int = ..., random_state: Any | None = ...): ...
    def entropy(self): ...

class dirichlet_gen(multi_rv_generic):
    __doc__: Any
    def __init__(self, seed: Any | None = ...) -> None: ...
    def __call__(self, alpha, seed: Any | None = ...): ...
    def logpdf(self, x, alpha): ...
    def pdf(self, x, alpha): ...
    def mean(self, alpha): ...
    def var(self, alpha): ...
    def entropy(self, alpha): ...
    def rvs(self, alpha, size: int = ..., random_state: Any | None = ...): ...

dirichlet: Any

class dirichlet_frozen(multi_rv_frozen):
    alpha: Any
    def __init__(self, alpha, seed: Any | None = ...) -> None: ...
    def logpdf(self, x): ...
    def pdf(self, x): ...
    def mean(self): ...
    def var(self): ...
    def entropy(self): ...
    def rvs(self, size: int = ..., random_state: Any | None = ...): ...

class wishart_gen(multi_rv_generic):
    __doc__: Any
    def __init__(self, seed: Any | None = ...) -> None: ...
    def __call__(
        self, df: Any | None = ..., scale: Any | None = ..., seed: Any | None = ...
    ): ...
    def logpdf(self, x, df, scale): ...
    def pdf(self, x, df, scale): ...
    def mean(self, df, scale): ...
    def mode(self, df, scale): ...
    def var(self, df, scale): ...
    def rvs(self, df, scale, size: int = ..., random_state: Any | None = ...): ...
    def entropy(self, df, scale): ...

wishart: Any

class wishart_frozen(multi_rv_frozen):
    def __init__(self, df, scale, seed: Any | None = ...) -> None: ...
    def logpdf(self, x): ...
    def pdf(self, x): ...
    def mean(self): ...
    def mode(self): ...
    def var(self): ...
    def rvs(self, size: int = ..., random_state: Any | None = ...): ...
    def entropy(self): ...

class invwishart_gen(wishart_gen):
    __doc__: Any
    def __init__(self, seed: Any | None = ...) -> None: ...
    def __call__(
        self, df: Any | None = ..., scale: Any | None = ..., seed: Any | None = ...
    ): ...
    def logpdf(self, x, df, scale): ...
    def pdf(self, x, df, scale): ...
    def mean(self, df, scale): ...
    def mode(self, df, scale): ...
    def var(self, df, scale): ...
    def rvs(self, df, scale, size: int = ..., random_state: Any | None = ...): ...
    def entropy(self, df, scale): ...

invwishart: Any

class invwishart_frozen(multi_rv_frozen):
    log_det_scale: Any
    inv_scale: Any
    C: Any
    def __init__(self, df, scale, seed: Any | None = ...) -> None: ...
    def logpdf(self, x): ...
    def pdf(self, x): ...
    def mean(self): ...
    def mode(self): ...
    def var(self): ...
    def rvs(self, size: int = ..., random_state: Any | None = ...): ...
    def entropy(self): ...

class multinomial_gen(multi_rv_generic):
    __doc__: Any
    def __init__(self, seed: Any | None = ...) -> None: ...
    def __call__(self, n, p, seed: Any | None = ...): ...
    def logpmf(self, x, n, p): ...
    def pmf(self, x, n, p): ...
    def mean(self, n, p): ...
    def cov(self, n, p): ...
    def entropy(self, n, p): ...
    def rvs(self, n, p, size: Any | None = ..., random_state: Any | None = ...): ...

multinomial: Any

class multinomial_frozen(multi_rv_frozen):
    def __init__(self, n, p, seed: Any | None = ...): ...
    def logpmf(self, x): ...
    def pmf(self, x): ...
    def mean(self): ...
    def cov(self): ...
    def entropy(self): ...
    def rvs(self, size: int = ..., random_state: Any | None = ...): ...

class special_ortho_group_gen(multi_rv_generic):
    __doc__: Any
    def __init__(self, seed: Any | None = ...) -> None: ...
    def __call__(self, dim: Any | None = ..., seed: Any | None = ...): ...
    def rvs(self, dim, size: int = ..., random_state: Any | None = ...): ...

special_ortho_group: Any

class special_ortho_group_frozen(multi_rv_frozen):
    dim: Any
    def __init__(self, dim: Any | None = ..., seed: Any | None = ...) -> None: ...
    def rvs(self, size: int = ..., random_state: Any | None = ...): ...

class ortho_group_gen(multi_rv_generic):
    __doc__: Any
    def __init__(self, seed: Any | None = ...) -> None: ...
    def __call__(self, dim: Any | None = ..., seed: Any | None = ...): ...
    def rvs(self, dim, size: int = ..., random_state: Any | None = ...): ...

ortho_group: Any

class ortho_group_frozen(multi_rv_frozen):
    dim: Any
    def __init__(self, dim: Any | None = ..., seed: Any | None = ...) -> None: ...
    def rvs(self, size: int = ..., random_state: Any | None = ...): ...

class random_correlation_gen(multi_rv_generic):
    __doc__: Any
    def __init__(self, seed: Any | None = ...) -> None: ...
    def __call__(
        self, eigs, seed: Any | None = ..., tol: float = ..., diag_tol: float = ...
    ): ...
    def rvs(
        self,
        eigs,
        random_state: Any | None = ...,
        tol: float = ...,
        diag_tol: float = ...,
    ): ...

random_correlation: Any

class random_correlation_frozen(multi_rv_frozen):
    tol: Any
    diag_tol: Any
    def __init__(
        self, eigs, seed: Any | None = ..., tol: float = ..., diag_tol: float = ...
    ) -> None: ...
    def rvs(self, random_state: Any | None = ...): ...

class unitary_group_gen(multi_rv_generic):
    __doc__: Any
    def __init__(self, seed: Any | None = ...) -> None: ...
    def __call__(self, dim: Any | None = ..., seed: Any | None = ...): ...
    def rvs(self, dim, size: int = ..., random_state: Any | None = ...): ...

unitary_group: Any

class unitary_group_frozen(multi_rv_frozen):
    dim: Any
    def __init__(self, dim: Any | None = ..., seed: Any | None = ...) -> None: ...
    def rvs(self, size: int = ..., random_state: Any | None = ...): ...

class multivariate_t_gen(multi_rv_generic):
    __doc__: Any
    def __init__(self, seed: Any | None = ...) -> None: ...
    def __call__(
        self,
        loc: Any | None = ...,
        shape: int = ...,
        df: int = ...,
        allow_singular: bool = ...,
        seed: Any | None = ...,
    ): ...
    def pdf(
        self,
        x,
        loc: Any | None = ...,
        shape: int = ...,
        df: int = ...,
        allow_singular: bool = ...,
    ): ...
    def logpdf(self, x, loc: Any | None = ..., shape: int = ..., df: int = ...): ...
    def cdf(
        self,
        x,
        loc: Any | None = ...,
        shape: int = ...,
        df: int = ...,
        allow_singular: bool = ...,
        *,
        maxpts: Any | None = ...,
        lower_limit: Any | None = ...,
        random_state: Any | None = ...
    ): ...
    def entropy(self, loc: Any | None = ..., shape: int = ..., df: int = ...): ...
    def rvs(
        self,
        loc: Any | None = ...,
        shape: int = ...,
        df: int = ...,
        size: int = ...,
        random_state: Any | None = ...,
    ): ...

class multivariate_t_frozen(multi_rv_frozen):
    shape_info: Any
    def __init__(
        self,
        loc: Any | None = ...,
        shape: int = ...,
        df: int = ...,
        allow_singular: bool = ...,
        seed: Any | None = ...,
    ) -> None: ...
    def logpdf(self, x): ...
    def cdf(
        self,
        x,
        *,
        maxpts: Any | None = ...,
        lower_limit: Any | None = ...,
        random_state: Any | None = ...
    ): ...
    def pdf(self, x): ...
    def rvs(self, size: int = ..., random_state: Any | None = ...): ...
    def entropy(self): ...

multivariate_t: Any

class multivariate_hypergeom_gen(multi_rv_generic):
    __doc__: Any
    def __init__(self, seed: Any | None = ...) -> None: ...
    def __call__(self, m, n, seed: Any | None = ...): ...
    def logpmf(self, x, m, n): ...
    def pmf(self, x, m, n): ...
    def mean(self, m, n): ...
    def var(self, m, n): ...
    def cov(self, m, n): ...
    def rvs(self, m, n, size: Any | None = ..., random_state: Any | None = ...): ...

multivariate_hypergeom: Any

class multivariate_hypergeom_frozen(multi_rv_frozen):
    def __init__(self, m, n, seed: Any | None = ...): ...
    def logpmf(self, x): ...
    def pmf(self, x): ...
    def mean(self): ...
    def var(self): ...
    def cov(self): ...
    def rvs(self, size: int = ..., random_state: Any | None = ...): ...

class random_table_gen(multi_rv_generic):
    def __init__(self, seed: Any | None = ...) -> None: ...
    def __call__(self, row, col, *, seed: Any | None = ...): ...
    def logpmf(self, x, row, col): ...
    def pmf(self, x, row, col): ...
    def mean(self, row, col): ...
    def rvs(
        self,
        row,
        col,
        *,
        size: Any | None = ...,
        method: Any | None = ...,
        random_state: Any | None = ...
    ): ...

random_table: Any

class random_table_frozen(multi_rv_frozen):
    def __init__(self, row, col, *, seed: Any | None = ...): ...
    def logpmf(self, x): ...
    def pmf(self, x): ...
    def mean(self): ...
    def rvs(
        self,
        size: Any | None = ...,
        method: Any | None = ...,
        random_state: Any | None = ...,
    ): ...

class uniform_direction_gen(multi_rv_generic):
    __doc__: Any
    def __init__(self, seed: Any | None = ...) -> None: ...
    def __call__(self, dim: Any | None = ..., seed: Any | None = ...): ...
    def rvs(self, dim, size: Any | None = ..., random_state: Any | None = ...): ...

uniform_direction: Any

class uniform_direction_frozen(multi_rv_frozen):
    dim: Any
    def __init__(self, dim: Any | None = ..., seed: Any | None = ...) -> None: ...
    def rvs(self, size: Any | None = ..., random_state: Any | None = ...): ...

class dirichlet_multinomial_gen(multi_rv_generic):
    __doc__: Any
    def __init__(self, seed: Any | None = ...) -> None: ...
    def __call__(self, alpha, n, seed: Any | None = ...): ...
    def logpmf(self, x, alpha, n): ...
    def pmf(self, x, alpha, n): ...
    def mean(self, alpha, n): ...
    def var(self, alpha, n): ...
    def cov(self, alpha, n): ...

dirichlet_multinomial: Any

class dirichlet_multinomial_frozen(multi_rv_frozen):
    alpha: Any
    n: Any
    def __init__(self, alpha, n, seed: Any | None = ...) -> None: ...
    def logpmf(self, x): ...
    def pmf(self, x): ...
    def mean(self): ...
    def var(self): ...
    def cov(self): ...

class vonmises_fisher_gen(multi_rv_generic):
    def __init__(self, seed: Any | None = ...) -> None: ...
    def __call__(
        self, mu: Any | None = ..., kappa: int = ..., seed: Any | None = ...
    ): ...
    def logpdf(self, x, mu: Any | None = ..., kappa: int = ...): ...
    def pdf(self, x, mu: Any | None = ..., kappa: int = ...): ...
    def rvs(
        self,
        mu: Any | None = ...,
        kappa: int = ...,
        size: int = ...,
        random_state: Any | None = ...,
    ): ...
    def entropy(self, mu: Any | None = ..., kappa: int = ...): ...
    def fit(self, x): ...

vonmises_fisher: Any

class vonmises_fisher_frozen(multi_rv_frozen):
    def __init__(
        self, mu: Any | None = ..., kappa: int = ..., seed: Any | None = ...
    ) -> None: ...
    def logpdf(self, x): ...
    def pdf(self, x): ...
    def rvs(self, size: int = ..., random_state: Any | None = ...): ...
    def entropy(self): ...
