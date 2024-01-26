from ._distn_infrastructure import rv_continuous
from scipy.stats._warnings_errors import FitError
from typing import Any

class ksone_gen(rv_continuous): ...

ksone: Any

class kstwo_gen(rv_continuous): ...

kstwo: Any

class kstwobign_gen(rv_continuous): ...

kstwobign: Any

class norm_gen(rv_continuous):
    def fit(self, data, **kwds): ...

norm: Any

class alpha_gen(rv_continuous): ...

alpha: Any

class anglit_gen(rv_continuous): ...

anglit: Any

class arcsine_gen(rv_continuous): ...

arcsine: Any

class FitDataError(ValueError):
    args: Any
    def __init__(self, distr, lower, upper) -> None: ...

class FitSolverError(FitError):
    args: Any
    def __init__(self, mesg) -> None: ...

class beta_gen(rv_continuous):
    def fit(self, data, *args, **kwds): ...

beta: Any

class betaprime_gen(rv_continuous): ...

betaprime: Any

class bradford_gen(rv_continuous): ...

bradford: Any

class burr_gen(rv_continuous): ...

burr: Any

class burr12_gen(rv_continuous): ...

burr12: Any

class fisk_gen(burr_gen): ...

fisk: Any

class cauchy_gen(rv_continuous): ...

cauchy: Any

class chi_gen(rv_continuous): ...

chi: Any

class chi2_gen(rv_continuous): ...

chi2: Any

class cosine_gen(rv_continuous): ...

cosine: Any

class dgamma_gen(rv_continuous): ...

dgamma: Any

class dweibull_gen(rv_continuous): ...

dweibull: Any

class expon_gen(rv_continuous):
    def fit(self, data, *args, **kwds): ...

expon: Any

class exponnorm_gen(rv_continuous): ...

exponnorm: Any

class exponweib_gen(rv_continuous): ...

exponweib: Any

class exponpow_gen(rv_continuous): ...

exponpow: Any

class fatiguelife_gen(rv_continuous): ...

fatiguelife: Any

class foldcauchy_gen(rv_continuous): ...

foldcauchy: Any

class f_gen(rv_continuous): ...

f: Any

class foldnorm_gen(rv_continuous): ...

foldnorm: Any

class weibull_min_gen(rv_continuous):
    def fit(self, data, *args, **kwds): ...

weibull_min: Any

class truncweibull_min_gen(rv_continuous): ...

truncweibull_min: Any

class weibull_max_gen(rv_continuous): ...

weibull_max: Any

class genlogistic_gen(rv_continuous): ...

genlogistic: Any

class genpareto_gen(rv_continuous): ...

genpareto: Any

class genexpon_gen(rv_continuous): ...

genexpon: Any

class genextreme_gen(rv_continuous): ...

genextreme: Any

class gamma_gen(rv_continuous):
    def fit(self, data, *args, **kwds): ...

gamma: Any

class erlang_gen(gamma_gen):
    def fit(self, data, *args, **kwds): ...

erlang: Any

class gengamma_gen(rv_continuous): ...

gengamma: Any

class genhalflogistic_gen(rv_continuous): ...

genhalflogistic: Any

class genhyperbolic_gen(rv_continuous): ...

genhyperbolic: Any

class gompertz_gen(rv_continuous): ...

gompertz: Any

class gumbel_r_gen(rv_continuous):
    def fit(self, data, *args, **kwds): ...

gumbel_r: Any

class gumbel_l_gen(rv_continuous):
    def fit(self, data, *args, **kwds): ...

gumbel_l: Any

class halfcauchy_gen(rv_continuous): ...

halfcauchy: Any

class halflogistic_gen(rv_continuous): ...

halflogistic: Any

class halfnorm_gen(rv_continuous): ...

halfnorm: Any

class hypsecant_gen(rv_continuous): ...

hypsecant: Any

class gausshyper_gen(rv_continuous): ...

gausshyper: Any

class invgamma_gen(rv_continuous): ...

invgamma: Any

class invgauss_gen(rv_continuous):
    def fit(self, data, *args, **kwds): ...

invgauss: Any

class geninvgauss_gen(rv_continuous): ...

geninvgauss: Any

class norminvgauss_gen(rv_continuous): ...

norminvgauss: Any

class invweibull_gen(rv_continuous): ...

invweibull: Any

class johnsonsb_gen(rv_continuous): ...

johnsonsb: Any

class johnsonsu_gen(rv_continuous): ...

johnsonsu: Any

class laplace_gen(rv_continuous):
    def fit(self, data, *args, **kwds): ...

laplace: Any

class laplace_asymmetric_gen(rv_continuous): ...

laplace_asymmetric: Any

class levy_gen(rv_continuous): ...

levy: Any

class levy_l_gen(rv_continuous): ...

levy_l: Any

class logistic_gen(rv_continuous):
    def fit(self, data, *args, **kwds): ...

logistic: Any

class loggamma_gen(rv_continuous): ...

loggamma: Any

class loglaplace_gen(rv_continuous): ...

loglaplace: Any

class lognorm_gen(rv_continuous):
    def fit(self, data, *args, **kwds): ...

lognorm: Any

class gibrat_gen(rv_continuous): ...

gibrat: Any

class maxwell_gen(rv_continuous): ...

maxwell: Any

class mielke_gen(rv_continuous): ...

mielke: Any

class kappa4_gen(rv_continuous): ...

kappa4: Any

class kappa3_gen(rv_continuous): ...

kappa3: Any

class moyal_gen(rv_continuous): ...

moyal: Any

class nakagami_gen(rv_continuous): ...

nakagami: Any

class ncx2_gen(rv_continuous): ...

ncx2: Any

class ncf_gen(rv_continuous): ...

ncf: Any

class t_gen(rv_continuous): ...

t: Any

class nct_gen(rv_continuous): ...

nct: Any

class pareto_gen(rv_continuous):
    def fit(self, data, *args, **kwds): ...

pareto: Any

class lomax_gen(rv_continuous): ...

lomax: Any

class pearson3_gen(rv_continuous):
    def fit(self, data, *args, **kwds): ...

pearson3: Any

class powerlaw_gen(rv_continuous):
    def fit(self, data, *args, **kwds): ...

powerlaw: Any

class powerlognorm_gen(rv_continuous): ...

powerlognorm: Any

class powernorm_gen(rv_continuous): ...

powernorm: Any

class rdist_gen(rv_continuous): ...

rdist: Any

class rayleigh_gen(rv_continuous):
    def fit(self, data, *args, **kwds): ...

rayleigh: Any

class reciprocal_gen(rv_continuous):
    fit_note: str
    def fit(self, data, *args, **kwds): ...

loguniform: Any
reciprocal: Any

class rice_gen(rv_continuous): ...

rice: Any

class recipinvgauss_gen(rv_continuous): ...

recipinvgauss: Any

class semicircular_gen(rv_continuous): ...

semicircular: Any

class skewcauchy_gen(rv_continuous): ...

skewcauchy: Any

class skewnorm_gen(rv_continuous):
    def fit(self, data, *args, **kwds): ...

skewnorm: Any

class trapezoid_gen(rv_continuous): ...

trapezoid: Any
trapz: Any

class triang_gen(rv_continuous): ...

triang: Any

class truncexpon_gen(rv_continuous): ...

truncexpon: Any

class truncnorm_gen(rv_continuous): ...

truncnorm: Any

class truncpareto_gen(rv_continuous):
    def fit(self, data, *args, **kwds): ...

truncpareto: Any

class tukeylambda_gen(rv_continuous): ...

tukeylambda: Any

class FitUniformFixedScaleDataError(FitDataError):
    args: Any
    def __init__(self, ptp, fscale) -> None: ...

class uniform_gen(rv_continuous):
    def fit(self, data, *args, **kwds): ...

uniform: Any

class vonmises_gen(rv_continuous):
    def rvs(self, *args, **kwds): ...
    def expect(
        self,
        func: Any | None = ...,
        args=...,
        loc: int = ...,
        scale: int = ...,
        lb: Any | None = ...,
        ub: Any | None = ...,
        conditional: bool = ...,
        **kwds
    ): ...
    def fit(self, data, *args, **kwds): ...

vonmises: Any
vonmises_line: Any

class wald_gen(invgauss_gen): ...

wald: Any

class wrapcauchy_gen(rv_continuous): ...

wrapcauchy: Any

class gennorm_gen(rv_continuous): ...

gennorm: Any

class halfgennorm_gen(rv_continuous): ...

halfgennorm: Any

class crystalball_gen(rv_continuous): ...

crystalball: Any

class argus_gen(rv_continuous): ...

argus: Any

class rv_histogram(rv_continuous):
    def __init__(
        self, histogram, *args, density: Any | None = ..., **kwargs
    ) -> None: ...

class studentized_range_gen(rv_continuous): ...

studentized_range: Any

class rel_breitwigner_gen(rv_continuous):
    def fit(self, data, *args, **kwds): ...

rel_breitwigner: Any
