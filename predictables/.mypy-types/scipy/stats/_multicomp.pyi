import numpy as np
import numpy.typing as npt
from scipy._lib._util import DecimalNumber, SeedType
from scipy.stats._common import ConfidenceInterval
from typing import Literal

class DunnettResult:
    statistic: np.ndarray
    pvalue: np.ndarray
    def confidence_interval(
        self, confidence_level: DecimalNumber = ...
    ) -> ConfidenceInterval: ...

def dunnett(
    *samples: npt.ArrayLike,
    control: npt.ArrayLike,
    alternative: Literal["two-sided", "less", "greater"] = ...,
    random_state: SeedType = ...
) -> DunnettResult: ...
