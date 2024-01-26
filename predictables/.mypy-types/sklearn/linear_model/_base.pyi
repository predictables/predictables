from ..base import (
    BaseEstimator as BaseEstimator,
    ClassifierMixin as ClassifierMixin,
    MultiOutputMixin as MultiOutputMixin,
    RegressorMixin as RegressorMixin,
)
from ..utils import check_array as check_array, check_random_state as check_random_state
from ..utils._array_api import get_namespace as get_namespace
from ..utils._seq_dataset import (
    ArrayDataset32 as ArrayDataset32,
    ArrayDataset64 as ArrayDataset64,
    CSRDataset32 as CSRDataset32,
    CSRDataset64 as CSRDataset64,
)
from ..utils.extmath import safe_sparse_dot as safe_sparse_dot
from ..utils.parallel import Parallel as Parallel, delayed as delayed
from ..utils.sparsefuncs import (
    inplace_column_scale as inplace_column_scale,
    mean_variance_axis as mean_variance_axis,
)
from ..utils.validation import (
    FLOAT_DTYPES as FLOAT_DTYPES,
    check_is_fitted as check_is_fitted,
)
from abc import ABCMeta, abstractmethod
from typing import Any

SPARSE_INTERCEPT_DECAY: float

def make_dataset(X, y, sample_weight, random_state: Any | None = ...): ...

class LinearModel(BaseEstimator, metaclass=ABCMeta):
    @abstractmethod
    def fit(self, X, y): ...
    def predict(self, X): ...

class LinearClassifierMixin(ClassifierMixin):
    def decision_function(self, X): ...
    def predict(self, X): ...

class SparseCoefMixin:
    coef_: Any
    def densify(self): ...
    def sparsify(self): ...

class LinearRegression(MultiOutputMixin, RegressorMixin, LinearModel):
    fit_intercept: Any
    copy_X: Any
    n_jobs: Any
    positive: Any
    def __init__(
        self,
        *,
        fit_intercept: bool = ...,
        copy_X: bool = ...,
        n_jobs: Any | None = ...,
        positive: bool = ...
    ) -> None: ...
    coef_: Any
    def fit(self, X, y, sample_weight: Any | None = ...): ...
