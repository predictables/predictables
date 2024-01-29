from ..._loss.loss import (
    BaseLoss as BaseLoss,
    HalfBinomialLoss as HalfBinomialLoss,
    HalfGammaLoss as HalfGammaLoss,
    HalfMultinomialLoss as HalfMultinomialLoss,
    HalfPoissonLoss as HalfPoissonLoss,
    PinballLoss as PinballLoss,
)
from ...base import (
    BaseEstimator as BaseEstimator,
    ClassifierMixin as ClassifierMixin,
    RegressorMixin as RegressorMixin,
    is_classifier as is_classifier,
)
from ...metrics import check_scoring as check_scoring
from ...model_selection import train_test_split as train_test_split
from ...preprocessing import LabelEncoder as LabelEncoder
from ...utils import (
    check_random_state as check_random_state,
    compute_sample_weight as compute_sample_weight,
    resample as resample,
)
from ...utils._param_validation import (
    Interval as Interval,
    RealNotInt as RealNotInt,
    StrOptions as StrOptions,
)
from ...utils.multiclass import (
    check_classification_targets as check_classification_targets,
)
from ...utils.validation import (
    check_consistent_length as check_consistent_length,
    check_is_fitted as check_is_fitted,
)
from .common import G_H_DTYPE as G_H_DTYPE, X_DTYPE as X_DTYPE, Y_DTYPE as Y_DTYPE
from .grower import TreeGrower as TreeGrower
from abc import ABC, abstractmethod
from collections.abc import Generator
from typing import Any

class BaseHistGradientBoosting(BaseEstimator, ABC):
    loss: Any
    learning_rate: Any
    max_iter: Any
    max_leaf_nodes: Any
    max_depth: Any
    min_samples_leaf: Any
    l2_regularization: Any
    max_bins: Any
    monotonic_cst: Any
    interaction_cst: Any
    categorical_features: Any
    warm_start: Any
    early_stopping: Any
    scoring: Any
    validation_fraction: Any
    n_iter_no_change: Any
    tol: Any
    verbose: Any
    random_state: Any
    @abstractmethod
    def __init__(
        self,
        loss,
        *,
        learning_rate,
        max_iter,
        max_leaf_nodes,
        max_depth,
        min_samples_leaf,
        l2_regularization,
        max_bins,
        categorical_features,
        monotonic_cst,
        interaction_cst,
        warm_start,
        early_stopping,
        scoring,
        validation_fraction,
        n_iter_no_change,
        tol,
        verbose,
        random_state
    ): ...
    do_early_stopping_: Any
    train_score_: Any
    validation_score_: Any
    def fit(self, X, y, sample_weight: Any | None = ...): ...
    @property
    def n_iter_(self): ...

class HistGradientBoostingRegressor(RegressorMixin, BaseHistGradientBoosting):
    quantile: Any
    def __init__(
        self,
        loss: str = ...,
        *,
        quantile: Any | None = ...,
        learning_rate: float = ...,
        max_iter: int = ...,
        max_leaf_nodes: int = ...,
        max_depth: Any | None = ...,
        min_samples_leaf: int = ...,
        l2_regularization: float = ...,
        max_bins: int = ...,
        categorical_features: Any | None = ...,
        monotonic_cst: Any | None = ...,
        interaction_cst: Any | None = ...,
        warm_start: bool = ...,
        early_stopping: str = ...,
        scoring: str = ...,
        validation_fraction: float = ...,
        n_iter_no_change: int = ...,
        tol: float = ...,
        verbose: int = ...,
        random_state: Any | None = ...
    ) -> None: ...
    def predict(self, X): ...
    def staged_predict(self, X) -> Generator[Any, None, None]: ...

class HistGradientBoostingClassifier(ClassifierMixin, BaseHistGradientBoosting):
    class_weight: Any
    def __init__(
        self,
        loss: str = ...,
        *,
        learning_rate: float = ...,
        max_iter: int = ...,
        max_leaf_nodes: int = ...,
        max_depth: Any | None = ...,
        min_samples_leaf: int = ...,
        l2_regularization: float = ...,
        max_bins: int = ...,
        categorical_features: Any | None = ...,
        monotonic_cst: Any | None = ...,
        interaction_cst: Any | None = ...,
        warm_start: bool = ...,
        early_stopping: str = ...,
        scoring: str = ...,
        validation_fraction: float = ...,
        n_iter_no_change: int = ...,
        tol: float = ...,
        verbose: int = ...,
        random_state: Any | None = ...,
        class_weight: Any | None = ...
    ) -> None: ...
    def predict(self, X): ...
    def staged_predict(self, X) -> Generator[Any, None, None]: ...
    def predict_proba(self, X): ...
    def staged_predict_proba(self, X) -> Generator[Any, None, None]: ...
    def decision_function(self, X): ...
    def staged_decision_function(self, X) -> Generator[Any, None, None]: ...
