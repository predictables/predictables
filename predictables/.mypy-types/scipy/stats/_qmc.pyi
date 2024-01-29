import numpy as np
import numpy.typing as npt
from abc import ABC, abstractmethod
from scipy._lib._util import DecimalNumber, IntNumber, SeedType
from typing import Any, ClassVar, Literal, overload

def scale(
    sample: npt.ArrayLike,
    l_bounds: npt.ArrayLike,
    u_bounds: npt.ArrayLike,
    *,
    reverse: bool = ...
) -> np.ndarray: ...
def discrepancy(
    sample: npt.ArrayLike,
    *,
    iterative: bool = ...,
    method: Literal["CD", "WD", "MD", "L2-star"] = ...,
    workers: IntNumber = ...
) -> float: ...
def update_discrepancy(
    x_new: npt.ArrayLike, sample: npt.ArrayLike, initial_disc: DecimalNumber
) -> float: ...

class QMCEngine(ABC):
    d: Any
    rng: Any
    rng_seed: Any
    num_generated: int
    optimization_method: Any
    @abstractmethod
    def __init__(
        self,
        d: IntNumber,
        *,
        optimization: Union[Literal["random-cd", "lloyd"], None] = ...,
        seed: SeedType = ...
    ): ...
    def random(self, n: IntNumber = ..., *, workers: IntNumber = ...) -> np.ndarray: ...
    def integers(
        self,
        l_bounds: npt.ArrayLike,
        *,
        u_bounds: Union[npt.ArrayLike, None] = ...,
        n: IntNumber = ...,
        endpoint: bool = ...,
        workers: IntNumber = ...
    ) -> np.ndarray: ...
    def reset(self) -> QMCEngine: ...
    def fast_forward(self, n: IntNumber) -> QMCEngine: ...

class Halton(QMCEngine):
    seed: Any
    base: Any
    scramble: Any
    def __init__(
        self,
        d: IntNumber,
        *,
        scramble: bool = ...,
        optimization: Union[Literal["random-cd", "lloyd"], None] = ...,
        seed: SeedType = ...
    ) -> None: ...

class LatinHypercube(QMCEngine):
    scramble: Any
    lhs_method: Any
    def __init__(
        self,
        d: IntNumber,
        *,
        centered: bool = ...,
        scramble: bool = ...,
        strength: int = ...,
        optimization: Union[Literal["random-cd", "lloyd"], None] = ...,
        seed: SeedType = ...
    ) -> None: ...

class Sobol(QMCEngine):
    MAXDIM: ClassVar[int]
    bits: Any
    dtype_i: Any
    maxn: Any
    def __init__(
        self,
        d: IntNumber,
        *,
        scramble: bool = ...,
        bits: Union[IntNumber, None] = ...,
        seed: SeedType = ...,
        optimization: Union[Literal["random-cd", "lloyd"], None] = ...
    ) -> None: ...
    def random_base2(self, m: IntNumber) -> np.ndarray: ...
    def reset(self) -> Sobol: ...
    def fast_forward(self, n: IntNumber) -> Sobol: ...

class PoissonDisk(QMCEngine):
    hypersphere_method: Any
    radius_factor: Any
    radius: Any
    radius_squared: Any
    ncandidates: Any
    cell_size: Any
    grid_size: Any
    def __init__(
        self,
        d: IntNumber,
        *,
        radius: DecimalNumber = ...,
        hypersphere: Literal["volume", "surface"] = ...,
        ncandidates: IntNumber = ...,
        optimization: Union[Literal["random-cd", "lloyd"], None] = ...,
        seed: SeedType = ...
    ) -> None: ...
    def fill_space(self) -> np.ndarray: ...
    def reset(self) -> PoissonDisk: ...

class MultivariateNormalQMC:
    engine: Any
    def __init__(
        self,
        mean: npt.ArrayLike,
        cov: Union[npt.ArrayLike, None] = ...,
        *,
        cov_root: Union[npt.ArrayLike, None] = ...,
        inv_transform: bool = ...,
        engine: Union[QMCEngine, None] = ...,
        seed: SeedType = ...
    ) -> None: ...
    def random(self, n: IntNumber = ...) -> np.ndarray: ...

class MultinomialQMC:
    pvals: Any
    n_trials: Any
    engine: Any
    def __init__(
        self,
        pvals: npt.ArrayLike,
        n_trials: IntNumber,
        *,
        engine: Union[QMCEngine, None] = ...,
        seed: SeedType = ...
    ) -> None: ...
    def random(self, n: IntNumber = ...) -> np.ndarray: ...
