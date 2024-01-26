from ..base import (
    BaseEstimator as BaseEstimator,
    ClassNamePrefixFeaturesOutMixin as ClassNamePrefixFeaturesOutMixin,
    TransformerMixin as TransformerMixin,
)
from ..utils.validation import check_is_fitted as check_is_fitted
from abc import ABCMeta, abstractmethod
from typing import Any

class _BasePCA(
    ClassNamePrefixFeaturesOutMixin, TransformerMixin, BaseEstimator, metaclass=ABCMeta
):
    def get_covariance(self): ...
    def get_precision(self): ...
    @abstractmethod
    def fit(self, X, y: Any | None = ...): ...
    def transform(self, X): ...
    def inverse_transform(self, X): ...
