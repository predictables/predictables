from ..exceptions import UndefinedMetricWarning as UndefinedMetricWarning
from ..preprocessing import (
    LabelBinarizer as LabelBinarizer,
    LabelEncoder as LabelEncoder,
)
from ..utils import (
    assert_all_finite as assert_all_finite,
    check_array as check_array,
    check_consistent_length as check_consistent_length,
    column_or_1d as column_or_1d,
)
from ..utils._param_validation import (
    Interval as Interval,
    Options as Options,
    StrOptions as StrOptions,
    validate_params as validate_params,
)
from ..utils.multiclass import (
    type_of_target as type_of_target,
    unique_labels as unique_labels,
)
from ..utils.sparsefuncs import count_nonzero as count_nonzero
from typing import Any

def accuracy_score(
    y_true, y_pred, *, normalize: bool = ..., sample_weight: Any | None = ...
): ...
def confusion_matrix(
    y_true,
    y_pred,
    *,
    labels: Any | None = ...,
    sample_weight: Any | None = ...,
    normalize: Any | None = ...
): ...
def multilabel_confusion_matrix(
    y_true,
    y_pred,
    *,
    sample_weight: Any | None = ...,
    labels: Any | None = ...,
    samplewise: bool = ...
): ...
def cohen_kappa_score(
    y1,
    y2,
    *,
    labels: Any | None = ...,
    weights: Any | None = ...,
    sample_weight: Any | None = ...
): ...
def jaccard_score(
    y_true,
    y_pred,
    *,
    labels: Any | None = ...,
    pos_label: int = ...,
    average: str = ...,
    sample_weight: Any | None = ...,
    zero_division: str = ...
): ...
def matthews_corrcoef(y_true, y_pred, *, sample_weight: Any | None = ...): ...
def zero_one_loss(
    y_true, y_pred, *, normalize: bool = ..., sample_weight: Any | None = ...
): ...
def f1_score(
    y_true,
    y_pred,
    *,
    labels: Any | None = ...,
    pos_label: int = ...,
    average: str = ...,
    sample_weight: Any | None = ...,
    zero_division: str = ...
): ...
def fbeta_score(
    y_true,
    y_pred,
    *,
    beta,
    labels: Any | None = ...,
    pos_label: int = ...,
    average: str = ...,
    sample_weight: Any | None = ...,
    zero_division: str = ...
): ...
def precision_recall_fscore_support(
    y_true,
    y_pred,
    *,
    beta: float = ...,
    labels: Any | None = ...,
    pos_label: int = ...,
    average: Any | None = ...,
    warn_for=...,
    sample_weight: Any | None = ...,
    zero_division: str = ...
): ...
def class_likelihood_ratios(
    y_true,
    y_pred,
    *,
    labels: Any | None = ...,
    sample_weight: Any | None = ...,
    raise_warning: bool = ...
): ...
def precision_score(
    y_true,
    y_pred,
    *,
    labels: Any | None = ...,
    pos_label: int = ...,
    average: str = ...,
    sample_weight: Any | None = ...,
    zero_division: str = ...
): ...
def recall_score(
    y_true,
    y_pred,
    *,
    labels: Any | None = ...,
    pos_label: int = ...,
    average: str = ...,
    sample_weight: Any | None = ...,
    zero_division: str = ...
): ...
def balanced_accuracy_score(
    y_true, y_pred, *, sample_weight: Any | None = ..., adjusted: bool = ...
): ...
def classification_report(
    y_true,
    y_pred,
    *,
    labels: Any | None = ...,
    target_names: Any | None = ...,
    sample_weight: Any | None = ...,
    digits: int = ...,
    output_dict: bool = ...,
    zero_division: str = ...
): ...
def hamming_loss(y_true, y_pred, *, sample_weight: Any | None = ...): ...
def log_loss(
    y_true,
    y_pred,
    *,
    eps: str = ...,
    normalize: bool = ...,
    sample_weight: Any | None = ...,
    labels: Any | None = ...
): ...
def hinge_loss(
    y_true, pred_decision, *, labels: Any | None = ..., sample_weight: Any | None = ...
): ...
def brier_score_loss(
    y_true, y_prob, *, sample_weight: Any | None = ..., pos_label: Any | None = ...
): ...
