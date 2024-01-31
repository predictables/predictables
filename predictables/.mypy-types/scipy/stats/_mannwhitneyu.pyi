from scipy import special as special, stats as stats
from typing import Any, NamedTuple

class _MWU:
    def __init__(self) -> None: ...
    def pmf(self, k, m, n): ...
    def pmf_recursive(self, k, m, n): ...
    def pmf_iterative(self, k, m, n): ...
    def cdf(self, k, m, n): ...
    def sf(self, k, m, n): ...

class MannwhitneyuResult(NamedTuple):
    statistic: Any
    pvalue: Any

def mannwhitneyu(
    x,
    y,
    use_continuity: bool = ...,
    alternative: str = ...,
    axis: int = ...,
    method: str = ...,
): ...