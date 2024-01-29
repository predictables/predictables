from ._common import ConfidenceInterval as ConfidenceInterval
from ._discrete_distns import nchypergeom_fisher as nchypergeom_fisher
from scipy.optimize import brentq as brentq
from scipy.special import ndtri as ndtri
from typing import Any

class OddsRatioResult:
    statistic: Any
    def __init__(self, _table, _kind, statistic) -> None: ...
    def confidence_interval(
        self, confidence_level: float = ..., alternative: str = ...
    ): ...

def odds_ratio(table, *, kind: str = ...): ...
