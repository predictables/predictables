from ..exceptions import UndefinedMetricWarning as UndefinedMetricWarning
from ..preprocessing import label_binarize as label_binarize
from ..utils import (
    assert_all_finite as assert_all_finite,
    check_array as check_array,
    check_consistent_length as check_consistent_length,
    column_or_1d as column_or_1d,
)
from ..utils._param_validation import (
    Interval as Interval,
    StrOptions as StrOptions,
    validate_params as validate_params,
)
from ..utils.extmath import stable_cumsum as stable_cumsum
from ..utils.fixes import trapezoid as trapezoid
from ..utils.multiclass import type_of_target as type_of_target
from ..utils.sparsefuncs import count_nonzero as count_nonzero
from typing import Any

def auc(x, y): ...
def average_precision_score(
    y_true,
    y_score,
    *,
    average: str = ...,
    pos_label: int = ...,
    sample_weight: Any | None = ...
): ...
def det_curve(
    y_true, y_score, pos_label: Any | None = ..., sample_weight: Any | None = ...
): ...
def roc_auc_score(
    y_true,
    y_score,
    *,
    average: str = ...,
    sample_weight: Any | None = ...,
    max_fpr: Any | None = ...,
    multi_class: str = ...,
    labels: Any | None = ...
): ...
def precision_recall_curve(
    y_true,
    probas_pred,
    *,
    pos_label: Any | None = ...,
    sample_weight: Any | None = ...,
    drop_intermediate: bool = ...
): ...
def roc_curve(
    y_true,
    y_score,
    *,
    pos_label: Any | None = ...,
    sample_weight: Any | None = ...,
    drop_intermediate: bool = ...
): ...
def label_ranking_average_precision_score(
    y_true, y_score, *, sample_weight: Any | None = ...
): ...
def coverage_error(y_true, y_score, *, sample_weight: Any | None = ...): ...
def label_ranking_loss(y_true, y_score, *, sample_weight: Any | None = ...): ...
def dcg_score(
    y_true,
    y_score,
    *,
    k: Any | None = ...,
    log_base: int = ...,
    sample_weight: Any | None = ...,
    ignore_ties: bool = ...
): ...
def ndcg_score(
    y_true,
    y_score,
    *,
    k: Any | None = ...,
    sample_weight: Any | None = ...,
    ignore_ties: bool = ...
): ...
def top_k_accuracy_score(
    y_true,
    y_score,
    *,
    k: int = ...,
    normalize: bool = ...,
    sample_weight: Any | None = ...,
    labels: Any | None = ...
): ...
