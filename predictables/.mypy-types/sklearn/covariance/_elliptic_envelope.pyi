from ..base import OutlierMixin as OutlierMixin
from ..metrics import accuracy_score as accuracy_score
from ..utils._param_validation import Interval as Interval
from ..utils.validation import check_is_fitted as check_is_fitted
from ._robust_covariance import MinCovDet as MinCovDet
from typing import Any

class EllipticEnvelope(OutlierMixin, MinCovDet):
    contamination: Any
    def __init__(
        self,
        *,
        store_precision: bool = ...,
        assume_centered: bool = ...,
        support_fraction: Any | None = ...,
        contamination: float = ...,
        random_state: Any | None = ...
    ) -> None: ...
    offset_: Any
    def fit(self, X, y: Any | None = ...): ...
    def decision_function(self, X): ...
    def score_samples(self, X): ...
    def predict(self, X): ...
    def score(self, X, y, sample_weight: Any | None = ...): ...
