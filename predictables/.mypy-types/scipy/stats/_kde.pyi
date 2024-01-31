from typing import Any

class gaussian_kde:
    dataset: Any
    def __init__(
        self, dataset, bw_method: Any | None = ..., weights: Any | None = ...
    ) -> None: ...
    def evaluate(self, points): ...
    __call__: Any
    def integrate_gaussian(self, mean, cov): ...
    def integrate_box_1d(self, low, high): ...
    def integrate_box(self, low_bounds, high_bounds, maxpts: Any | None = ...): ...
    def integrate_kde(self, other): ...
    def resample(self, size: Any | None = ..., seed: Any | None = ...): ...
    def scotts_factor(self): ...
    def silverman_factor(self): ...
    covariance_factor: Any
    def set_bandwidth(self, bw_method: Any | None = ...): ...
    factor: Any
    @property
    def inv_cov(self): ...
    def pdf(self, x): ...
    def logpdf(self, x): ...
    def marginal(self, dimensions): ...
    @property
    def weights(self): ...
    @property
    def neff(self): ...