from ...utils import (
    check_matplotlib_support as check_matplotlib_support,
    check_random_state as check_random_state,
)
from typing import Any

class PredictionErrorDisplay:
    y_true: Any
    y_pred: Any
    def __init__(self, *, y_true, y_pred) -> None: ...
    line_: Any
    scatter_: Any
    ax_: Any
    figure_: Any
    def plot(
        self,
        ax: Any | None = ...,
        *,
        kind: str = ...,
        scatter_kwargs: Any | None = ...,
        line_kwargs: Any | None = ...
    ): ...
    @classmethod
    def from_estimator(
        cls,
        estimator,
        X,
        y,
        *,
        kind: str = ...,
        subsample: int = ...,
        random_state: Any | None = ...,
        ax: Any | None = ...,
        scatter_kwargs: Any | None = ...,
        line_kwargs: Any | None = ...
    ): ...
    @classmethod
    def from_predictions(
        cls,
        y_true,
        y_pred,
        *,
        kind: str = ...,
        subsample: int = ...,
        random_state: Any | None = ...,
        ax: Any | None = ...,
        scatter_kwargs: Any | None = ...,
        line_kwargs: Any | None = ...
    ): ...
