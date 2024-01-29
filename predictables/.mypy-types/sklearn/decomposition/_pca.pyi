from ..utils import check_random_state as check_random_state
from ..utils._param_validation import (
    Interval as Interval,
    RealNotInt as RealNotInt,
    StrOptions as StrOptions,
)
from ..utils.deprecation import deprecated as deprecated
from ..utils.extmath import (
    fast_logdet as fast_logdet,
    randomized_svd as randomized_svd,
    stable_cumsum as stable_cumsum,
    svd_flip as svd_flip,
)
from ..utils.validation import check_is_fitted as check_is_fitted
from ._base import _BasePCA
from typing import Any

class PCA(_BasePCA):
    n_components: Any
    copy: Any
    whiten: Any
    svd_solver: Any
    tol: Any
    iterated_power: Any
    n_oversamples: Any
    power_iteration_normalizer: Any
    random_state: Any
    def __init__(
        self,
        n_components: Any | None = ...,
        *,
        copy: bool = ...,
        whiten: bool = ...,
        svd_solver: str = ...,
        tol: float = ...,
        iterated_power: str = ...,
        n_oversamples: int = ...,
        power_iteration_normalizer: str = ...,
        random_state: Any | None = ...
    ) -> None: ...
    @property
    def n_features_(self): ...
    def fit(self, X, y: Any | None = ...): ...
    def fit_transform(self, X, y: Any | None = ...): ...
    def score_samples(self, X): ...
    def score(self, X, y: Any | None = ...): ...
