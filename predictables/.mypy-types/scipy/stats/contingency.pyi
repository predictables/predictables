from ._crosstab import crosstab as crosstab
from ._odds_ratio import odds_ratio as odds_ratio
from ._relative_risk import relative_risk as relative_risk
from typing import Any

def margins(a): ...
def expected_freq(observed): ...
def chi2_contingency(observed, correction: bool = ..., lambda_: Any | None = ...): ...
def association(
    observed, method: str = ..., correction: bool = ..., lambda_: Any | None = ...
): ...