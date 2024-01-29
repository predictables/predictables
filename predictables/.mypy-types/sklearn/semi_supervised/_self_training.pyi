from ..base import BaseEstimator, MetaEstimatorMixin
from typing import Any

class SelfTrainingClassifier(MetaEstimatorMixin, BaseEstimator):
    base_estimator: Any
    threshold: Any
    criterion: Any
    k_best: Any
    max_iter: Any
    verbose: Any
    def __init__(
        self,
        base_estimator,
        threshold: float = ...,
        criterion: str = ...,
        k_best: int = ...,
        max_iter: int = ...,
        verbose: bool = ...,
    ) -> None: ...
    base_estimator_: Any
    transduction_: Any
    labeled_iter_: Any
    n_iter_: int
    termination_condition_: str
    classes_: Any
    def fit(self, X, y): ...
    def predict(self, X): ...
    def predict_proba(self, X): ...
    def decision_function(self, X): ...
    def predict_log_proba(self, X): ...
    def score(self, X, y): ...
