from ..base import (
    BaseEstimator,
    ClassNamePrefixFeaturesOutMixin,
    MultiOutputMixin,
    RegressorMixin,
    TransformerMixin,
)
from abc import ABCMeta, abstractmethod
from typing import Any

class _PLS(
    ClassNamePrefixFeaturesOutMixin,
    TransformerMixin,
    RegressorMixin,
    MultiOutputMixin,
    BaseEstimator,
    metaclass=ABCMeta,
):
    n_components: Any
    deflation_mode: Any
    mode: Any
    scale: Any
    algorithm: Any
    max_iter: Any
    tol: Any
    copy: Any
    @abstractmethod
    def __init__(
        self,
        n_components: int = ...,
        *,
        scale: bool = ...,
        deflation_mode: str = ...,
        mode: str = ...,
        algorithm: str = ...,
        max_iter: int = ...,
        tol: float = ...,
        copy: bool = ...
    ): ...
    x_weights_: Any
    y_weights_: Any
    x_loadings_: Any
    y_loadings_: Any
    n_iter_: Any
    x_rotations_: Any
    y_rotations_: Any
    coef_: Any
    intercept_: Any
    def fit(self, X, Y): ...
    def transform(self, X, Y: Any | None = ..., copy: bool = ...): ...
    def inverse_transform(self, X, Y: Any | None = ...): ...
    def predict(self, X, copy: bool = ...): ...
    def fit_transform(self, X, y: Any | None = ...): ...

class PLSRegression(_PLS):
    def __init__(
        self,
        n_components: int = ...,
        *,
        scale: bool = ...,
        max_iter: int = ...,
        tol: float = ...,
        copy: bool = ...
    ) -> None: ...
    x_scores_: Any
    y_scores_: Any
    def fit(self, X, Y): ...

class PLSCanonical(_PLS):
    def __init__(
        self,
        n_components: int = ...,
        *,
        scale: bool = ...,
        algorithm: str = ...,
        max_iter: int = ...,
        tol: float = ...,
        copy: bool = ...
    ) -> None: ...

class CCA(_PLS):
    def __init__(
        self,
        n_components: int = ...,
        *,
        scale: bool = ...,
        max_iter: int = ...,
        tol: float = ...,
        copy: bool = ...
    ) -> None: ...

class PLSSVD(ClassNamePrefixFeaturesOutMixin, TransformerMixin, BaseEstimator):
    n_components: Any
    scale: Any
    copy: Any
    def __init__(
        self, n_components: int = ..., *, scale: bool = ..., copy: bool = ...
    ) -> None: ...
    x_weights_: Any
    y_weights_: Any
    def fit(self, X, Y): ...
    def transform(self, X, Y: Any | None = ...): ...
    def fit_transform(self, X, y: Any | None = ...): ...
