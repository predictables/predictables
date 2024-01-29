from typing import Any, NamedTuple

def argstoarray(*args): ...
def find_repeats(arr): ...
def count_tied_groups(x, use_missing: bool = ...): ...
def rankdata(data, axis: Any | None = ..., use_missing: bool = ...): ...

class ModeResult(NamedTuple):
    mode: Any
    count: Any

def mode(a, axis: int = ...): ...
def msign(x): ...
def pearsonr(x, y): ...
def spearmanr(
    x,
    y: Any | None = ...,
    use_ties: bool = ...,
    axis: Any | None = ...,
    nan_policy: str = ...,
    alternative: str = ...,
): ...
def kendalltau(
    x,
    y,
    use_ties: bool = ...,
    use_missing: bool = ...,
    method: str = ...,
    alternative: str = ...,
): ...
def kendalltau_seasonal(x): ...

class PointbiserialrResult(NamedTuple):
    correlation: Any
    pvalue: Any

def pointbiserialr(x, y): ...
def linregress(x, y: Any | None = ...): ...
def theilslopes(y, x: Any | None = ..., alpha: float = ..., method: str = ...): ...
def siegelslopes(y, x: Any | None = ..., method: str = ...): ...
def sen_seasonal_slopes(x): ...

class Ttest_1sampResult(NamedTuple):
    statistic: Any
    pvalue: Any

def ttest_1samp(a, popmean, axis: int = ..., alternative: str = ...): ...

ttest_onesamp = ttest_1samp

class Ttest_indResult(NamedTuple):
    statistic: Any
    pvalue: Any

def ttest_ind(a, b, axis: int = ..., equal_var: bool = ..., alternative: str = ...): ...

class Ttest_relResult(NamedTuple):
    statistic: Any
    pvalue: Any

def ttest_rel(a, b, axis: int = ..., alternative: str = ...): ...

class MannwhitneyuResult(NamedTuple):
    statistic: Any
    pvalue: Any

def mannwhitneyu(x, y, use_continuity: bool = ...): ...

class KruskalResult(NamedTuple):
    statistic: Any
    pvalue: Any

def kruskal(*args): ...

kruskalwallis = kruskal

def ks_1samp(x, cdf, args=..., alternative: str = ..., method: str = ...): ...
def ks_2samp(data1, data2, alternative: str = ..., method: str = ...): ...

ks_twosamp = ks_2samp

def kstest(data1, data2, args=..., alternative: str = ..., method: str = ...): ...
def trima(a, limits: Any | None = ..., inclusive=...): ...
def trimr(a, limits: Any | None = ..., inclusive=..., axis: Any | None = ...): ...
def trim(
    a,
    limits: Any | None = ...,
    inclusive=...,
    relative: bool = ...,
    axis: Any | None = ...,
): ...
def trimboth(
    data, proportiontocut: float = ..., inclusive=..., axis: Any | None = ...
): ...
def trimtail(
    data,
    proportiontocut: float = ...,
    tail: str = ...,
    inclusive=...,
    axis: Any | None = ...,
): ...

trim1 = trimtail

def trimmed_mean(
    a, limits=..., inclusive=..., relative: bool = ..., axis: Any | None = ...
): ...
def trimmed_var(
    a,
    limits=...,
    inclusive=...,
    relative: bool = ...,
    axis: Any | None = ...,
    ddof: int = ...,
): ...
def trimmed_std(
    a,
    limits=...,
    inclusive=...,
    relative: bool = ...,
    axis: Any | None = ...,
    ddof: int = ...,
): ...
def trimmed_stde(a, limits=..., inclusive=..., axis: Any | None = ...): ...
def tmean(a, limits: Any | None = ..., inclusive=..., axis: Any | None = ...): ...
def tvar(
    a, limits: Any | None = ..., inclusive=..., axis: int = ..., ddof: int = ...
): ...
def tmin(a, lowerlimit: Any | None = ..., axis: int = ..., inclusive: bool = ...): ...
def tmax(a, upperlimit: Any | None = ..., axis: int = ..., inclusive: bool = ...): ...
def tsem(
    a, limits: Any | None = ..., inclusive=..., axis: int = ..., ddof: int = ...
): ...
def winsorize(
    a,
    limits: Any | None = ...,
    inclusive=...,
    inplace: bool = ...,
    axis: Any | None = ...,
    nan_policy: str = ...,
): ...
def moment(a, moment: int = ..., axis: int = ...): ...
def variation(a, axis: int = ..., ddof: int = ...): ...
def skew(a, axis: int = ..., bias: bool = ...): ...
def kurtosis(a, axis: int = ..., fisher: bool = ..., bias: bool = ...): ...

class DescribeResult(NamedTuple):
    nobs: Any
    minmax: Any
    mean: Any
    variance: Any
    skewness: Any
    kurtosis: Any

def describe(a, axis: int = ..., ddof: int = ..., bias: bool = ...): ...

class SkewtestResult(NamedTuple):
    statistic: Any
    pvalue: Any

def skewtest(a, axis: int = ..., alternative: str = ...): ...

class KurtosistestResult(NamedTuple):
    statistic: Any
    pvalue: Any

def kurtosistest(a, axis: int = ..., alternative: str = ...): ...

class NormaltestResult(NamedTuple):
    statistic: Any
    pvalue: Any

def normaltest(a, axis: int = ...): ...
def mquantiles(
    a,
    prob=...,
    alphap: float = ...,
    betap: float = ...,
    axis: Any | None = ...,
    limit=...,
): ...
def scoreatpercentile(
    data, per, limit=..., alphap: float = ..., betap: float = ...
): ...
def plotting_positions(data, alpha: float = ..., beta: float = ...): ...

meppf = plotting_positions

def obrientransform(*args): ...
def sem(a, axis: int = ..., ddof: int = ...): ...

class F_onewayResult(NamedTuple):
    statistic: Any
    pvalue: Any

def f_oneway(*args): ...

class FriedmanchisquareResult(NamedTuple):
    statistic: Any
    pvalue: Any

def friedmanchisquare(*args): ...

class BrunnerMunzelResult(NamedTuple):
    statistic: Any
    pvalue: Any

def brunnermunzel(x, y, alternative: str = ..., distribution: str = ...): ...
