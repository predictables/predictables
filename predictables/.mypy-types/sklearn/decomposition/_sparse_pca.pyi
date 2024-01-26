from ..base import (
    BaseEstimator as BaseEstimator,
    ClassNamePrefixFeaturesOutMixin as ClassNamePrefixFeaturesOutMixin,
    TransformerMixin as TransformerMixin,
)
from ..linear_model import ridge_regression as ridge_regression
from ..utils import check_random_state as check_random_state
from ..utils._param_validation import (
    Hidden as Hidden,
    Interval as Interval,
    StrOptions as StrOptions,
)
from ..utils.extmath import svd_flip as svd_flip
from ..utils.validation import (
    check_array as check_array,
    check_is_fitted as check_is_fitted,
)
from ._dict_learning import (
    MiniBatchDictionaryLearning as MiniBatchDictionaryLearning,
    dict_learning as dict_learning,
)
from typing import Any

class _BaseSparsePCA(ClassNamePrefixFeaturesOutMixin, TransformerMixin, BaseEstimator):
    n_components: Any
    alpha: Any
    ridge_alpha: Any
    max_iter: Any
    tol: Any
    method: Any
    n_jobs: Any
    verbose: Any
    random_state: Any
    def __init__(
        self,
        n_components: Any | None = ...,
        *,
        alpha: int = ...,
        ridge_alpha: float = ...,
        max_iter: int = ...,
        tol: float = ...,
        method: str = ...,
        n_jobs: Any | None = ...,
        verbose: bool = ...,
        random_state: Any | None = ...
    ) -> None: ...
    mean_: Any
    def fit(self, X, y: Any | None = ...): ...
    def transform(self, X): ...
    def inverse_transform(self, X): ...

class SparsePCA(_BaseSparsePCA):
    U_init: Any
    V_init: Any
    def __init__(
        self,
        n_components: Any | None = ...,
        *,
        alpha: int = ...,
        ridge_alpha: float = ...,
        max_iter: int = ...,
        tol: float = ...,
        method: str = ...,
        n_jobs: Any | None = ...,
        U_init: Any | None = ...,
        V_init: Any | None = ...,
        verbose: bool = ...,
        random_state: Any | None = ...
    ) -> None: ...

class MiniBatchSparsePCA(_BaseSparsePCA):
    n_iter: Any
    callback: Any
    batch_size: Any
    shuffle: Any
    max_no_improvement: Any
    def __init__(
        self,
        n_components: Any | None = ...,
        *,
        alpha: int = ...,
        ridge_alpha: float = ...,
        n_iter: str = ...,
        max_iter: Any | None = ...,
        callback: Any | None = ...,
        batch_size: int = ...,
        verbose: bool = ...,
        shuffle: bool = ...,
        n_jobs: Any | None = ...,
        method: str = ...,
        random_state: Any | None = ...,
        tol: float = ...,
        max_no_improvement: int = ...
    ) -> None: ...
