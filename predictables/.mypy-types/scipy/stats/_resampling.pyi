import numpy as np
from ._common import ConfidenceInterval
from typing import Any

class BootstrapResult:
    confidence_interval: ConfidenceInterval
    bootstrap_distribution: np.ndarray
    standard_error: Union[float, np.ndarray]

def bootstrap(
    data,
    statistic,
    *,
    n_resamples: int = ...,
    batch: Any | None = ...,
    vectorized: Any | None = ...,
    paired: bool = ...,
    axis: int = ...,
    confidence_level: float = ...,
    alternative: str = ...,
    method: str = ...,
    bootstrap_result: Any | None = ...,
    random_state: Any | None = ...
): ...

class MonteCarloTestResult:
    statistic: Union[float, np.ndarray]
    pvalue: Union[float, np.ndarray]
    null_distribution: np.ndarray

def monte_carlo_test(
    data,
    rvs,
    statistic,
    *,
    vectorized: Any | None = ...,
    n_resamples: int = ...,
    batch: Any | None = ...,
    alternative: str = ...,
    axis: int = ...
): ...

class PermutationTestResult:
    statistic: Union[float, np.ndarray]
    pvalue: Union[float, np.ndarray]
    null_distribution: np.ndarray

def permutation_test(
    data,
    statistic,
    *,
    permutation_type: str = ...,
    vectorized: Any | None = ...,
    n_resamples: int = ...,
    batch: Any | None = ...,
    alternative: str = ...,
    axis: int = ...,
    random_state: Any | None = ...
): ...

class ResamplingMethod:
    n_resamples: int
    batch: int

class MonteCarloMethod(ResamplingMethod):
    rvs: object

class PermutationMethod(ResamplingMethod):
    random_state: object

class BootstrapMethod(ResamplingMethod):
    random_state: object
    method: str
