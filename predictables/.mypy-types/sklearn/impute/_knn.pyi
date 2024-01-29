from ..metrics import pairwise_distances_chunked as pairwise_distances_chunked
from ..utils import is_scalar_nan as is_scalar_nan
from ..utils._param_validation import (
    Hidden as Hidden,
    Interval as Interval,
    StrOptions as StrOptions,
)
from ..utils.validation import (
    FLOAT_DTYPES as FLOAT_DTYPES,
    check_is_fitted as check_is_fitted,
)
from ._base import _BaseImputer
from typing import Any

class KNNImputer(_BaseImputer):
    n_neighbors: Any
    weights: Any
    metric: Any
    copy: Any
    def __init__(
        self,
        *,
        missing_values=...,
        n_neighbors: int = ...,
        weights: str = ...,
        metric: str = ...,
        copy: bool = ...,
        add_indicator: bool = ...,
        keep_empty_features: bool = ...
    ) -> None: ...
    def fit(self, X, y: Any | None = ...): ...
    def transform(self, X): ...
    def get_feature_names_out(self, input_features: Any | None = ...): ...
