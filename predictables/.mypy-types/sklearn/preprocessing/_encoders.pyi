from ..base import BaseEstimator, OneToOneFeatureMixin, TransformerMixin
from typing import Any

class _BaseEncoder(TransformerMixin, BaseEstimator):
    @property
    def infrequent_categories_(self): ...

class OneHotEncoder(_BaseEncoder):
    categories: Any
    sparse: Any
    sparse_output: Any
    dtype: Any
    handle_unknown: Any
    drop: Any
    min_frequency: Any
    max_categories: Any
    feature_name_combiner: Any
    def __init__(
        self,
        *,
        categories: str = ...,
        drop: Any | None = ...,
        sparse: str = ...,
        sparse_output: bool = ...,
        dtype=...,
        handle_unknown: str = ...,
        min_frequency: Any | None = ...,
        max_categories: Any | None = ...,
        feature_name_combiner: str = ...
    ) -> None: ...
    def fit(self, X, y: Any | None = ...): ...
    def transform(self, X): ...
    def inverse_transform(self, X): ...
    def get_feature_names_out(self, input_features: Any | None = ...): ...

class OrdinalEncoder(OneToOneFeatureMixin, _BaseEncoder):
    categories: Any
    dtype: Any
    handle_unknown: Any
    unknown_value: Any
    encoded_missing_value: Any
    min_frequency: Any
    max_categories: Any
    def __init__(
        self,
        *,
        categories: str = ...,
        dtype=...,
        handle_unknown: str = ...,
        unknown_value: Any | None = ...,
        encoded_missing_value=...,
        min_frequency: Any | None = ...,
        max_categories: Any | None = ...
    ) -> None: ...
    def fit(self, X, y: Any | None = ...): ...
    def transform(self, X): ...
    def inverse_transform(self, X): ...
