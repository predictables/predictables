from ._common import ConfidenceInterval as ConfidenceInterval
from scipy.special import ndtri as ndtri

class RelativeRiskResult:
    relative_risk: float
    exposed_cases: int
    exposed_total: int
    control_cases: int
    control_total: int
    def confidence_interval(self, confidence_level: float = ...): ...

def relative_risk(exposed_cases, exposed_total, control_cases, control_total): ...
