from ...base import is_regressor as is_regressor
from ...preprocessing import LabelEncoder as LabelEncoder
from ...utils import check_matplotlib_support as check_matplotlib_support
from ...utils.validation import check_is_fitted as check_is_fitted
from typing import Any

class DecisionBoundaryDisplay:
    xx0: Any
    xx1: Any
    response: Any
    xlabel: Any
    ylabel: Any
    def __init__(
        self, *, xx0, xx1, response, xlabel: Any | None = ..., ylabel: Any | None = ...
    ) -> None: ...
    surface_: Any
    ax_: Any
    figure_: Any
    def plot(
        self,
        plot_method: str = ...,
        ax: Any | None = ...,
        xlabel: Any | None = ...,
        ylabel: Any | None = ...,
        **kwargs
    ): ...
    @classmethod
    def from_estimator(
        cls,
        estimator,
        X,
        *,
        grid_resolution: int = ...,
        eps: float = ...,
        plot_method: str = ...,
        response_method: str = ...,
        xlabel: Any | None = ...,
        ylabel: Any | None = ...,
        ax: Any | None = ...,
        **kwargs
    ): ...
