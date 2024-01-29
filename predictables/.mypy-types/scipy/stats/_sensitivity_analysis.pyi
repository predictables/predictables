import numpy as np
import numpy.typing as npt
from scipy._lib._util import DecimalNumber, IntNumber, SeedType
from scipy.stats._resampling import BootstrapResult
from typing import Callable, Literal, Protocol

class BootstrapSobolResult:
    first_order: BootstrapResult
    total_order: BootstrapResult

class SobolResult:
    first_order: np.ndarray
    total_order: np.ndarray
    def bootstrap(
        self, confidence_level: DecimalNumber = ..., n_resamples: IntNumber = ...
    ) -> BootstrapSobolResult: ...

class PPFDist(Protocol):
    @property
    def ppf(self) -> Callable[..., float]: ...

def sobol_indices(
    *,
    func: Union[
        Callable[[np.ndarray], npt.ArrayLike],
        dict[Literal["f_A", "f_B", "f_AB"], np.ndarray],
    ],
    n: IntNumber,
    dists: Union[list[PPFDist], None] = ...,
    method: Union[Callable, Literal["saltelli_2010"]] = ...,
    random_state: SeedType = ...
) -> SobolResult: ...
