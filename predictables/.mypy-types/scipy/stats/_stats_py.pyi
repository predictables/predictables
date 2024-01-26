from ._stats_mstats_common import (
    linregress as linregress,
    siegelslopes as siegelslopes,
    theilslopes as theilslopes,
)
from typing import Any, NamedTuple

def gmean(a, axis: int = ..., dtype: Any | None = ..., weights: Any | None = ...): ...
def hmean(
    a, axis: int = ..., dtype: Any | None = ..., *, weights: Any | None = ...
): ...
def pmean(
    a, p, *, axis: int = ..., dtype: Any | None = ..., weights: Any | None = ...
): ...

class ModeResult(NamedTuple):
    mode: Any
    count: Any

def mode(a, axis: int = ..., nan_policy: str = ..., keepdims: bool = ...): ...
def tmean(a, limits: Any | None = ..., inclusive=..., axis: Any | None = ...): ...
def tvar(
    a, limits: Any | None = ..., inclusive=..., axis: int = ..., ddof: int = ...
): ...
def tmin(
    a,
    lowerlimit: Any | None = ...,
    axis: int = ...,
    inclusive: bool = ...,
    nan_policy: str = ...,
): ...
def tmax(
    a,
    upperlimit: Any | None = ...,
    axis: int = ...,
    inclusive: bool = ...,
    nan_policy: str = ...,
): ...
def tstd(
    a, limits: Any | None = ..., inclusive=..., axis: int = ..., ddof: int = ...
): ...
def tsem(
    a, limits: Any | None = ..., inclusive=..., axis: int = ..., ddof: int = ...
): ...
def moment(
    a,
    moment: int = ...,
    axis: int = ...,
    nan_policy: str = ...,
    *,
    center: Any | None = ...
): ...
def skew(a, axis: int = ..., bias: bool = ..., nan_policy: str = ...): ...
def kurtosis(
    a, axis: int = ..., fisher: bool = ..., bias: bool = ..., nan_policy: str = ...
): ...

class DescribeResult(NamedTuple):
    nobs: Any
    minmax: Any
    mean: Any
    variance: Any
    skewness: Any
    kurtosis: Any

def describe(
    a, axis: int = ..., ddof: int = ..., bias: bool = ..., nan_policy: str = ...
): ...

class SkewtestResult(NamedTuple):
    statistic: Any
    pvalue: Any

def skewtest(a, axis: int = ..., nan_policy: str = ..., alternative: str = ...): ...

class KurtosistestResult(NamedTuple):
    statistic: Any
    pvalue: Any

def kurtosistest(a, axis: int = ..., nan_policy: str = ..., alternative: str = ...): ...

class NormaltestResult(NamedTuple):
    statistic: Any
    pvalue: Any

def normaltest(a, axis: int = ..., nan_policy: str = ...): ...
def jarque_bera(x, *, axis: Any | None = ...): ...
def scoreatpercentile(
    a, per, limit=..., interpolation_method: str = ..., axis: Any | None = ...
): ...
def percentileofscore(a, score, kind: str = ..., nan_policy: str = ...): ...

class HistogramResult(NamedTuple):
    count: Any
    lowerlimit: Any
    binsize: Any
    extrapoints: Any

class CumfreqResult(NamedTuple):
    cumcount: Any
    lowerlimit: Any
    binsize: Any
    extrapoints: Any

def cumfreq(
    a,
    numbins: int = ...,
    defaultreallimits: Any | None = ...,
    weights: Any | None = ...,
): ...

class RelfreqResult(NamedTuple):
    frequency: Any
    lowerlimit: Any
    binsize: Any
    extrapoints: Any

def relfreq(
    a,
    numbins: int = ...,
    defaultreallimits: Any | None = ...,
    weights: Any | None = ...,
): ...
def obrientransform(*samples): ...
def sem(a, axis: int = ..., ddof: int = ..., nan_policy: str = ...): ...
def zscore(a, axis: int = ..., ddof: int = ..., nan_policy: str = ...): ...
def gzscore(a, *, axis: int = ..., ddof: int = ..., nan_policy: str = ...): ...
def zmap(scores, compare, axis: int = ..., ddof: int = ..., nan_policy: str = ...): ...
def gstd(a, axis: int = ..., ddof: int = ...): ...
def iqr(
    x,
    axis: Any | None = ...,
    rng=...,
    scale: float = ...,
    nan_policy: str = ...,
    interpolation: str = ...,
    keepdims: bool = ...,
): ...
def median_abs_deviation(
    x, axis: int = ..., center=..., scale: float = ..., nan_policy: str = ...
): ...

class SigmaclipResult(NamedTuple):
    clipped: Any
    lower: Any
    upper: Any

def sigmaclip(a, low: float = ..., high: float = ...): ...
def trimboth(a, proportiontocut, axis: int = ...): ...
def trim1(a, proportiontocut, tail: str = ..., axis: int = ...): ...
def trim_mean(a, proportiontocut, axis: int = ...): ...

class F_onewayResult(NamedTuple):
    statistic: Any
    pvalue: Any

def f_oneway(*samples, axis: int = ...): ...
def alexandergovern(*samples, nan_policy: str = ...): ...

class AlexanderGovernResult:
    statistic: float
    pvalue: float

class ConfidenceInterval(NamedTuple):
    low: Any
    high: Any

class PearsonRResult(PearsonRResultBase):
    correlation: Any
    def __init__(self, statistic, pvalue, alternative, n, x, y) -> None: ...
    def confidence_interval(
        self, confidence_level: float = ..., method: Any | None = ...
    ): ...

def pearsonr(x, y, *, alternative: str = ..., method: Any | None = ...): ...
def fisher_exact(table, alternative: str = ...): ...
def spearmanr(
    a,
    b: Any | None = ...,
    axis: int = ...,
    nan_policy: str = ...,
    alternative: str = ...,
): ...
def pointbiserialr(x, y): ...
def kendalltau(
    x,
    y,
    initial_lexsort: Any | None = ...,
    nan_policy: str = ...,
    method: str = ...,
    variant: str = ...,
    alternative: str = ...,
): ...
def weightedtau(
    x, y, rank: bool = ..., weigher: Any | None = ..., additive: bool = ...
): ...

class _ParallelP:
    x: Any
    y: Any
    random_states: Any
    def __init__(self, x, y, random_states) -> None: ...
    def __call__(self, index): ...

def multiscale_graphcorr(
    x,
    y,
    compute_distance=...,
    reps: int = ...,
    workers: int = ...,
    is_twosamp: bool = ...,
    random_state: Any | None = ...,
): ...

class TtestResult(TtestResultBase):
    def __init__(
        self, statistic, pvalue, df, alternative, standard_error, estimate
    ) -> None: ...
    def confidence_interval(self, confidence_level: float = ...): ...

def ttest_1samp(
    a, popmean, axis: int = ..., nan_policy: str = ..., alternative: str = ...
): ...

class Ttest_indResult(NamedTuple):
    statistic: Any
    pvalue: Any

def ttest_ind_from_stats(
    mean1,
    std1,
    nobs1,
    mean2,
    std2,
    nobs2,
    equal_var: bool = ...,
    alternative: str = ...,
): ...
def ttest_ind(
    a,
    b,
    axis: int = ...,
    equal_var: bool = ...,
    nan_policy: str = ...,
    permutations: Any | None = ...,
    random_state: Any | None = ...,
    alternative: str = ...,
    trim: int = ...,
): ...
def ttest_rel(a, b, axis: int = ..., nan_policy: str = ..., alternative: str = ...): ...

class Power_divergenceResult(NamedTuple):
    statistic: Any
    pvalue: Any

def power_divergence(
    f_obs,
    f_exp: Any | None = ...,
    ddof: int = ...,
    axis: int = ...,
    lambda_: Any | None = ...,
): ...
def chisquare(f_obs, f_exp: Any | None = ..., ddof: int = ..., axis: int = ...): ...
def ks_1samp(x, cdf, args=..., alternative: str = ..., method: str = ...): ...

Ks_2sampResult = KstestResult

def ks_2samp(data1, data2, alternative: str = ..., method: str = ...): ...
def kstest(
    rvs, cdf, args=..., N: int = ..., alternative: str = ..., method: str = ...
): ...
def tiecorrect(rankvals): ...

class RanksumsResult(NamedTuple):
    statistic: Any
    pvalue: Any

def ranksums(x, y, alternative: str = ...): ...

class KruskalResult(NamedTuple):
    statistic: Any
    pvalue: Any

def kruskal(*samples, nan_policy: str = ...): ...

class FriedmanchisquareResult(NamedTuple):
    statistic: Any
    pvalue: Any

def friedmanchisquare(*samples): ...

class BrunnerMunzelResult(NamedTuple):
    statistic: Any
    pvalue: Any

def brunnermunzel(
    x, y, alternative: str = ..., distribution: str = ..., nan_policy: str = ...
): ...
def combine_pvalues(pvalues, method: str = ..., weights: Any | None = ...): ...
def wasserstein_distance(
    u_values, v_values, u_weights: Any | None = ..., v_weights: Any | None = ...
): ...
def energy_distance(
    u_values, v_values, u_weights: Any | None = ..., v_weights: Any | None = ...
): ...

class RepeatedResults(NamedTuple):
    values: Any
    counts: Any

def find_repeats(arr): ...
def rankdata(
    a, method: str = ..., *, axis: Any | None = ..., nan_policy: str = ...
): ...
def expectile(a, alpha: float = ..., *, weights: Any | None = ...): ...
