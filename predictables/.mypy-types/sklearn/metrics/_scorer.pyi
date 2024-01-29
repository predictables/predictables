from . import (
    accuracy_score as accuracy_score,
    average_precision_score as average_precision_score,
    balanced_accuracy_score as balanced_accuracy_score,
    brier_score_loss as brier_score_loss,
    class_likelihood_ratios as class_likelihood_ratios,
    explained_variance_score as explained_variance_score,
    f1_score as f1_score,
    jaccard_score as jaccard_score,
    log_loss as log_loss,
    matthews_corrcoef as matthews_corrcoef,
    max_error as max_error,
    mean_absolute_error as mean_absolute_error,
    mean_absolute_percentage_error as mean_absolute_percentage_error,
    mean_gamma_deviance as mean_gamma_deviance,
    mean_poisson_deviance as mean_poisson_deviance,
    mean_squared_error as mean_squared_error,
    mean_squared_log_error as mean_squared_log_error,
    median_absolute_error as median_absolute_error,
    precision_score as precision_score,
    r2_score as r2_score,
    recall_score as recall_score,
    roc_auc_score as roc_auc_score,
    top_k_accuracy_score as top_k_accuracy_score,
)
from ..base import is_regressor as is_regressor
from ..utils import Bunch as Bunch
from ..utils._param_validation import (
    HasMethods as HasMethods,
    StrOptions as StrOptions,
    validate_params as validate_params,
)
from ..utils.metadata_routing import (
    MetadataRequest as MetadataRequest,
    MetadataRouter as MetadataRouter,
    _MetadataRequester,
    get_routing_for_object as get_routing_for_object,
    process_routing as process_routing,
)
from ..utils.multiclass import type_of_target as type_of_target
from .cluster import (
    adjusted_mutual_info_score as adjusted_mutual_info_score,
    adjusted_rand_score as adjusted_rand_score,
    completeness_score as completeness_score,
    fowlkes_mallows_score as fowlkes_mallows_score,
    homogeneity_score as homogeneity_score,
    mutual_info_score as mutual_info_score,
    normalized_mutual_info_score as normalized_mutual_info_score,
    rand_score as rand_score,
    v_measure_score as v_measure_score,
)
from typing import Any

class _MultimetricScorer:
    def __init__(self, *, scorers, raise_exc: bool = ...) -> None: ...
    def __call__(self, estimator, *args, **kwargs): ...
    def get_metadata_routing(self): ...

class _BaseScorer(_MetadataRequester):
    def __init__(self, score_func, sign, kwargs) -> None: ...
    def __call__(
        self, estimator, X, y_true, sample_weight: Any | None = ..., **kwargs
    ): ...
    def set_score_request(self, **kwargs): ...

class _PredictScorer(_BaseScorer): ...
class _ProbaScorer(_BaseScorer): ...
class _ThresholdScorer(_BaseScorer): ...

def get_scorer(scoring): ...

class _PassthroughScorer:
    def __init__(self, estimator) -> None: ...
    def __call__(self, estimator, *args, **kwargs): ...
    def get_metadata_routing(self): ...

def make_scorer(
    score_func,
    *,
    greater_is_better: bool = ...,
    needs_proba: bool = ...,
    needs_threshold: bool = ...,
    **kwargs
): ...

explained_variance_scorer: Any
r2_scorer: Any
max_error_scorer: Any
neg_mean_squared_error_scorer: Any
neg_mean_squared_log_error_scorer: Any
neg_mean_absolute_error_scorer: Any
neg_mean_absolute_percentage_error_scorer: Any
neg_median_absolute_error_scorer: Any
neg_root_mean_squared_error_scorer: Any
neg_mean_poisson_deviance_scorer: Any
neg_mean_gamma_deviance_scorer: Any
accuracy_scorer: Any
balanced_accuracy_scorer: Any
matthews_corrcoef_scorer: Any

def positive_likelihood_ratio(y_true, y_pred): ...
def negative_likelihood_ratio(y_true, y_pred): ...

positive_likelihood_ratio_scorer: Any
neg_negative_likelihood_ratio_scorer: Any
top_k_accuracy_scorer: Any
roc_auc_scorer: Any
average_precision_scorer: Any
roc_auc_ovo_scorer: Any
roc_auc_ovo_weighted_scorer: Any
roc_auc_ovr_scorer: Any
roc_auc_ovr_weighted_scorer: Any
neg_log_loss_scorer: Any
neg_brier_score_scorer: Any
brier_score_loss_scorer: Any
adjusted_rand_scorer: Any
rand_scorer: Any
homogeneity_scorer: Any
completeness_scorer: Any
v_measure_scorer: Any
mutual_info_scorer: Any
adjusted_mutual_info_scorer: Any
normalized_mutual_info_scorer: Any
fowlkes_mallows_scorer: Any

def get_scorer_names(): ...

qualified_name: Any

def check_scoring(estimator, scoring: Any | None = ..., *, allow_none: bool = ...): ...
