from .base import (
    BaseEstimator as BaseEstimator,
    ClassifierMixin as ClassifierMixin,
    MultiOutputMixin as MultiOutputMixin,
    RegressorMixin as RegressorMixin,
)
from .utils import check_random_state as check_random_state
from .utils._param_validation import Interval as Interval, StrOptions as StrOptions
from .utils.multiclass import class_distribution as class_distribution
from .utils.validation import (
    check_array as check_array,
    check_consistent_length as check_consistent_length,
    check_is_fitted as check_is_fitted,
)
from typing import Any

class DummyClassifier(MultiOutputMixin, ClassifierMixin, BaseEstimator):
    strategy: Any
    random_state: Any
    constant: Any
    def __init__(
        self,
        *,
        strategy: str = ...,
        random_state: Any | None = ...,
        constant: Any | None = ...
    ) -> None: ...
    sparse_output_: Any
    n_outputs_: Any
    n_classes_: Any
    classes_: Any
    class_prior_: Any
    def fit(self, X, y, sample_weight: Any | None = ...): ...
    def predict(self, X): ...
    def predict_proba(self, X): ...
    def predict_log_proba(self, X): ...
    def score(self, X, y, sample_weight: Any | None = ...): ...

class DummyRegressor(MultiOutputMixin, RegressorMixin, BaseEstimator):
    strategy: Any
    constant: Any
    quantile: Any
    def __init__(
        self,
        *,
        strategy: str = ...,
        constant: Any | None = ...,
        quantile: Any | None = ...
    ) -> None: ...
    n_outputs_: Any
    constant_: Any
    def fit(self, X, y, sample_weight: Any | None = ...): ...
    def predict(self, X, return_std: bool = ...): ...
    def score(self, X, y, sample_weight: Any | None = ...): ...
