import numpy as np
from typing import Any, NamedTuple

class Epps_Singleton_2sampResult(NamedTuple):
    statistic: Any
    pvalue: Any

def epps_singleton_2samp(x, y, t=...): ...
def poisson_means_test(k1, n1, k2, n2, *, diff: int = ..., alternative: str = ...): ...

class CramerVonMisesResult:
    statistic: Any
    pvalue: Any
    def __init__(self, statistic, pvalue) -> None: ...

def cramervonmises(rvs, cdf, args=...): ...

class SomersDResult:
    statistic: float
    pvalue: float
    table: np.ndarray

def somersd(x, y: Any | None = ..., alternative: str = ...): ...

class BarnardExactResult:
    statistic: float
    pvalue: float

def barnard_exact(table, alternative: str = ..., pooled: bool = ..., n: int = ...): ...

class BoschlooExactResult:
    statistic: float
    pvalue: float

def boschloo_exact(table, alternative: str = ..., n: int = ...): ...
def cramervonmises_2samp(x, y, method: str = ...): ...

class TukeyHSDResult:
    statistic: Any
    pvalue: Any
    def __init__(self, statistic, pvalue, _nobs, _ntreatments, _stand_err) -> None: ...
    def confidence_interval(self, confidence_level: float = ...): ...

def tukey_hsd(*args): ...
