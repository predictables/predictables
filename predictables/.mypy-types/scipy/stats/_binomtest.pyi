from ._common import ConfidenceInterval as ConfidenceInterval
from ._discrete_distns import binom as binom
from scipy.optimize import brentq as brentq
from scipy.special import ndtri as ndtri
from typing import Any

class BinomTestResult:
    k: Any
    n: Any
    alternative: Any
    statistic: Any
    pvalue: Any
    proportion_estimate: Any
    def __init__(self, k, n, alternative, statistic, pvalue) -> None: ...
    def proportion_ci(self, confidence_level: float = ..., method: str = ...): ...

def binomtest(k, n, p: float = ..., alternative: str = ...): ...
