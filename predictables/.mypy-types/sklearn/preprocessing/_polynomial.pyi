from ..base import BaseEstimator, TransformerMixin
from typing import Any

class PolynomialFeatures(TransformerMixin, BaseEstimator):
    degree: Any
    interaction_only: Any
    include_bias: Any
    order: Any
    def __init__(
        self,
        degree: int = ...,
        *,
        interaction_only: bool = ...,
        include_bias: bool = ...,
        order: str = ...
    ) -> None: ...
    @property
    def powers_(self): ...
    def get_feature_names_out(self, input_features: Any | None = ...): ...
    n_output_features_: Any
    def fit(self, X, y: Any | None = ...): ...
    def transform(self, X): ...

class SplineTransformer(TransformerMixin, BaseEstimator):
    n_knots: Any
    degree: Any
    knots: Any
    extrapolation: Any
    include_bias: Any
    order: Any
    sparse_output: Any
    def __init__(
        self,
        n_knots: int = ...,
        degree: int = ...,
        *,
        knots: str = ...,
        extrapolation: str = ...,
        include_bias: bool = ...,
        order: str = ...,
        sparse_output: bool = ...
    ) -> None: ...
    def get_feature_names_out(self, input_features: Any | None = ...): ...
    bsplines_: Any
    n_features_out_: Any
    def fit(self, X, y: Any | None = ..., sample_weight: Any | None = ...): ...
    def transform(self, X): ...
