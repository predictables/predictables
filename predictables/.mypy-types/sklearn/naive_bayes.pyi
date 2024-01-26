from .base import BaseEstimator, ClassifierMixin
from abc import ABCMeta
from typing import Any

class _BaseNB(ClassifierMixin, BaseEstimator, metaclass=ABCMeta):
    def predict_joint_log_proba(self, X): ...
    def predict(self, X): ...
    def predict_log_proba(self, X): ...
    def predict_proba(self, X): ...

class GaussianNB(_BaseNB):
    priors: Any
    var_smoothing: Any
    def __init__(
        self, *, priors: Any | None = ..., var_smoothing: float = ...
    ) -> None: ...
    def fit(self, X, y, sample_weight: Any | None = ...): ...
    def partial_fit(
        self, X, y, classes: Any | None = ..., sample_weight: Any | None = ...
    ): ...

class _BaseDiscreteNB(_BaseNB):
    alpha: Any
    fit_prior: Any
    class_prior: Any
    force_alpha: Any
    def __init__(
        self,
        alpha: float = ...,
        fit_prior: bool = ...,
        class_prior: Any | None = ...,
        force_alpha: str = ...,
    ) -> None: ...
    def partial_fit(
        self, X, y, classes: Any | None = ..., sample_weight: Any | None = ...
    ): ...
    classes_: Any
    def fit(self, X, y, sample_weight: Any | None = ...): ...

class MultinomialNB(_BaseDiscreteNB):
    def __init__(
        self,
        *,
        alpha: float = ...,
        force_alpha: str = ...,
        fit_prior: bool = ...,
        class_prior: Any | None = ...
    ) -> None: ...

class ComplementNB(_BaseDiscreteNB):
    norm: Any
    def __init__(
        self,
        *,
        alpha: float = ...,
        force_alpha: str = ...,
        fit_prior: bool = ...,
        class_prior: Any | None = ...,
        norm: bool = ...
    ) -> None: ...

class BernoulliNB(_BaseDiscreteNB):
    binarize: Any
    def __init__(
        self,
        *,
        alpha: float = ...,
        force_alpha: str = ...,
        binarize: float = ...,
        fit_prior: bool = ...,
        class_prior: Any | None = ...
    ) -> None: ...

class CategoricalNB(_BaseDiscreteNB):
    min_categories: Any
    def __init__(
        self,
        *,
        alpha: float = ...,
        force_alpha: str = ...,
        fit_prior: bool = ...,
        class_prior: Any | None = ...,
        min_categories: Any | None = ...
    ) -> None: ...
    def fit(self, X, y, sample_weight: Any | None = ...): ...
    def partial_fit(
        self, X, y, classes: Any | None = ..., sample_weight: Any | None = ...
    ): ...
