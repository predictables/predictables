import numpy as np
import numpy.typing as npt
from scipy.stats._censored_data import CensoredData
from typing import Any, Literal

class EmpiricalDistributionFunction:
    quantiles: np.ndarray
    probabilities: np.ndarray
    def __init__(self, q, p, n, d, kind) -> None: ...
    def evaluate(self, x): ...
    def plot(self, ax: Any | None = ..., **matplotlib_kwargs): ...
    def confidence_interval(
        self, confidence_level: float = ..., *, method: str = ...
    ): ...

class ECDFResult:
    cdf: EmpiricalDistributionFunction
    sf: EmpiricalDistributionFunction
    def __init__(self, q, cdf, sf, n, d) -> None: ...

def ecdf(sample: Union[npt.ArrayLike, CensoredData]) -> ECDFResult: ...

class LogRankResult:
    statistic: np.ndarray
    pvalue: np.ndarray

def logrank(
    x: Union[npt.ArrayLike, CensoredData],
    y: Union[npt.ArrayLike, CensoredData],
    alternative: Literal["two-sided", "less", "greater"] = ...,
) -> LogRankResult: ...
