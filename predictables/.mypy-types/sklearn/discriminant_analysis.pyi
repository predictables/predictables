from .base import (
    BaseEstimator,
    ClassNamePrefixFeaturesOutMixin,
    ClassifierMixin,
    TransformerMixin,
)
from .linear_model._base import LinearClassifierMixin
from typing import Any

class LinearDiscriminantAnalysis(
    ClassNamePrefixFeaturesOutMixin,
    LinearClassifierMixin,
    TransformerMixin,
    BaseEstimator,
):
    solver: Any
    shrinkage: Any
    priors: Any
    n_components: Any
    store_covariance: Any
    tol: Any
    covariance_estimator: Any
    def __init__(
        self,
        solver: str = ...,
        shrinkage: Any | None = ...,
        priors: Any | None = ...,
        n_components: Any | None = ...,
        store_covariance: bool = ...,
        tol: float = ...,
        covariance_estimator: Any | None = ...,
    ) -> None: ...
    classes_: Any
    priors_: Any
    coef_: Any
    intercept_: Any
    def fit(self, X, y): ...
    def transform(self, X): ...
    def predict_proba(self, X): ...
    def predict_log_proba(self, X): ...
    def decision_function(self, X): ...

class QuadraticDiscriminantAnalysis(ClassifierMixin, BaseEstimator):
    priors: Any
    reg_param: Any
    store_covariance: Any
    tol: Any
    def __init__(
        self,
        *,
        priors: Any | None = ...,
        reg_param: float = ...,
        store_covariance: bool = ...,
        tol: float = ...
    ) -> None: ...
    priors_: Any
    covariance_: Any
    means_: Any
    scalings_: Any
    rotations_: Any
    def fit(self, X, y): ...
    def decision_function(self, X): ...
    def predict(self, X): ...
    def predict_proba(self, X): ...
    def predict_log_proba(self, X): ...
