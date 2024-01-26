from .base import BaseEstimator, RegressorMixin, TransformerMixin
from typing import Any

def check_increasing(x, y): ...
def isotonic_regression(
    y,
    *,
    sample_weight: Any | None = ...,
    y_min: Any | None = ...,
    y_max: Any | None = ...,
    increasing: bool = ...
): ...

class IsotonicRegression(RegressorMixin, TransformerMixin, BaseEstimator):
    y_min: Any
    y_max: Any
    increasing: Any
    out_of_bounds: Any
    def __init__(
        self,
        *,
        y_min: Any | None = ...,
        y_max: Any | None = ...,
        increasing: bool = ...,
        out_of_bounds: str = ...
    ) -> None: ...
    def fit(self, X, y, sample_weight: Any | None = ...): ...
    def transform(self, T): ...
    def predict(self, T): ...
    def get_feature_names_out(self, input_features: Any | None = ...): ...
