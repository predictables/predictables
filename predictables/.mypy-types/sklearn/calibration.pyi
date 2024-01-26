from .base import (
    BaseEstimator as BaseEstimator,
    ClassifierMixin as ClassifierMixin,
    MetaEstimatorMixin as MetaEstimatorMixin,
    RegressorMixin as RegressorMixin,
    clone as clone,
)
from .isotonic import IsotonicRegression as IsotonicRegression
from .model_selection import (
    check_cv as check_cv,
    cross_val_predict as cross_val_predict,
)
from .preprocessing import (
    LabelEncoder as LabelEncoder,
    label_binarize as label_binarize,
)
from .svm import LinearSVC as LinearSVC
from .utils import column_or_1d as column_or_1d, indexable as indexable
from .utils._param_validation import (
    HasMethods as HasMethods,
    Hidden as Hidden,
    Interval as Interval,
    StrOptions as StrOptions,
    validate_params as validate_params,
)
from .utils._plotting import _BinaryClassifierCurveDisplayMixin
from .utils.metadata_routing import (
    MetadataRouter as MetadataRouter,
    MethodMapping as MethodMapping,
    process_routing as process_routing,
)
from .utils.multiclass import (
    check_classification_targets as check_classification_targets,
)
from .utils.parallel import Parallel as Parallel, delayed as delayed
from .utils.validation import (
    check_consistent_length as check_consistent_length,
    check_is_fitted as check_is_fitted,
)
from sklearn.utils import Bunch as Bunch
from typing import Any

class CalibratedClassifierCV(ClassifierMixin, MetaEstimatorMixin, BaseEstimator):
    estimator: Any
    method: Any
    cv: Any
    n_jobs: Any
    ensemble: Any
    base_estimator: Any
    def __init__(
        self,
        estimator: Any | None = ...,
        *,
        method: str = ...,
        cv: Any | None = ...,
        n_jobs: Any | None = ...,
        ensemble: bool = ...,
        base_estimator: str = ...
    ) -> None: ...
    calibrated_classifiers_: Any
    classes_: Any
    n_features_in_: Any
    feature_names_in_: Any
    def fit(self, X, y, sample_weight: Any | None = ..., **fit_params): ...
    def predict_proba(self, X): ...
    def predict(self, X): ...
    def get_metadata_routing(self): ...

class _CalibratedClassifier:
    estimator: Any
    calibrators: Any
    classes: Any
    method: Any
    def __init__(
        self, estimator, calibrators, *, classes, method: str = ...
    ) -> None: ...
    def predict_proba(self, X): ...

class _SigmoidCalibration(RegressorMixin, BaseEstimator):
    def fit(self, X, y, sample_weight: Any | None = ...): ...
    def predict(self, T): ...

def calibration_curve(
    y_true,
    y_prob,
    *,
    pos_label: Any | None = ...,
    n_bins: int = ...,
    strategy: str = ...
): ...

class CalibrationDisplay(_BinaryClassifierCurveDisplayMixin):
    prob_true: Any
    prob_pred: Any
    y_prob: Any
    estimator_name: Any
    pos_label: Any
    def __init__(
        self,
        prob_true,
        prob_pred,
        y_prob,
        *,
        estimator_name: Any | None = ...,
        pos_label: Any | None = ...
    ) -> None: ...
    line_: Any
    def plot(
        self,
        *,
        ax: Any | None = ...,
        name: Any | None = ...,
        ref_line: bool = ...,
        **kwargs
    ): ...
    @classmethod
    def from_estimator(
        cls,
        estimator,
        X,
        y,
        *,
        n_bins: int = ...,
        strategy: str = ...,
        pos_label: Any | None = ...,
        name: Any | None = ...,
        ref_line: bool = ...,
        ax: Any | None = ...,
        **kwargs
    ): ...
    @classmethod
    def from_predictions(
        cls,
        y_true,
        y_prob,
        *,
        n_bins: int = ...,
        strategy: str = ...,
        pos_label: Any | None = ...,
        name: Any | None = ...,
        ref_line: bool = ...,
        ax: Any | None = ...,
        **kwargs
    ): ...
