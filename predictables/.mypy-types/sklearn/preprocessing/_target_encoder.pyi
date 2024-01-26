from ..base import OneToOneFeatureMixin as OneToOneFeatureMixin
from ..utils._param_validation import Interval as Interval, StrOptions as StrOptions
from ..utils.multiclass import type_of_target as type_of_target
from ..utils.validation import check_consistent_length as check_consistent_length
from ._encoders import _BaseEncoder
from typing import Any

class TargetEncoder(OneToOneFeatureMixin, _BaseEncoder):
    categories: Any
    smooth: Any
    target_type: Any
    cv: Any
    shuffle: Any
    random_state: Any
    def __init__(
        self,
        categories: str = ...,
        target_type: str = ...,
        smooth: str = ...,
        cv: int = ...,
        shuffle: bool = ...,
        random_state: Any | None = ...,
    ) -> None: ...
    def fit(self, X, y): ...
    def fit_transform(self, X, y): ...
    def transform(self, X): ...
