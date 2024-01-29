from ..base import OutlierMixin
from ._bagging import BaseBagging
from typing import Any

class IsolationForest(OutlierMixin, BaseBagging):
    contamination: Any
    def __init__(
        self,
        *,
        n_estimators: int = ...,
        max_samples: str = ...,
        contamination: str = ...,
        max_features: float = ...,
        bootstrap: bool = ...,
        n_jobs: Any | None = ...,
        random_state: Any | None = ...,
        verbose: int = ...,
        warm_start: bool = ...
    ) -> None: ...
    max_samples_: Any
    offset_: Any
    def fit(self, X, y: Any | None = ..., sample_weight: Any | None = ...): ...
    def predict(self, X): ...
    def decision_function(self, X): ...
    def score_samples(self, X): ...
