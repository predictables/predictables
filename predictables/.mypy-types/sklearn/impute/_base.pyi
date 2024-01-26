from ..base import BaseEstimator as BaseEstimator, TransformerMixin as TransformerMixin
from ..utils import is_scalar_nan as is_scalar_nan
from ..utils._param_validation import (
    MissingValues as MissingValues,
    StrOptions as StrOptions,
)
from ..utils.validation import (
    FLOAT_DTYPES as FLOAT_DTYPES,
    check_is_fitted as check_is_fitted,
)
from typing import Any

class _BaseImputer(TransformerMixin, BaseEstimator):
    missing_values: Any
    add_indicator: Any
    keep_empty_features: Any
    def __init__(
        self,
        *,
        missing_values=...,
        add_indicator: bool = ...,
        keep_empty_features: bool = ...
    ) -> None: ...

class SimpleImputer(_BaseImputer):
    strategy: Any
    fill_value: Any
    copy: Any
    def __init__(
        self,
        *,
        missing_values=...,
        strategy: str = ...,
        fill_value: Any | None = ...,
        copy: bool = ...,
        add_indicator: bool = ...,
        keep_empty_features: bool = ...
    ) -> None: ...
    statistics_: Any
    def fit(self, X, y: Any | None = ...): ...
    def transform(self, X): ...
    def inverse_transform(self, X): ...
    def get_feature_names_out(self, input_features: Any | None = ...): ...

class MissingIndicator(TransformerMixin, BaseEstimator):
    missing_values: Any
    features: Any
    sparse: Any
    error_on_new: Any
    def __init__(
        self,
        *,
        missing_values=...,
        features: str = ...,
        sparse: str = ...,
        error_on_new: bool = ...
    ) -> None: ...
    def fit(self, X, y: Any | None = ...): ...
    def transform(self, X): ...
    def fit_transform(self, X, y: Any | None = ...): ...
    def get_feature_names_out(self, input_features: Any | None = ...): ...
