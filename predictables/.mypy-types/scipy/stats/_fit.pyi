from scipy import optimize as optimize, stats as stats
from scipy._lib._util import check_random_state as check_random_state
from typing import Any, NamedTuple

class FitResult:
    discrete: Any
    pxf: Any
    params: Any
    success: Any
    message: Any
    def __init__(self, dist, data, discrete, res) -> None: ...
    def nllf(self, params: Any | None = ..., data: Any | None = ...): ...
    def plot(self, ax: Any | None = ..., *, plot_type: str = ...): ...

def fit(
    dist,
    data,
    bounds: Any | None = ...,
    *,
    guess: Any | None = ...,
    method: str = ...,
    optimizer=...
): ...

class GoodnessOfFitResult(NamedTuple):
    fit_result: Any
    statistic: Any
    pvalue: Any
    null_distribution: Any

def goodness_of_fit(
    dist,
    data,
    *,
    known_params: Any | None = ...,
    fit_params: Any | None = ...,
    guessed_params: Any | None = ...,
    statistic: str = ...,
    n_mc_samples: int = ...,
    random_state: Any | None = ...
): ...
