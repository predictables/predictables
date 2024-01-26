from ...utils._plotting import _BinaryClassifierCurveDisplayMixin
from .._ranking import (
    average_precision_score as average_precision_score,
    precision_recall_curve as precision_recall_curve,
)
from typing import Any

class PrecisionRecallDisplay(_BinaryClassifierCurveDisplayMixin):
    estimator_name: Any
    precision: Any
    recall: Any
    average_precision: Any
    pos_label: Any
    prevalence_pos_label: Any
    def __init__(
        self,
        precision,
        recall,
        *,
        average_precision: Any | None = ...,
        estimator_name: Any | None = ...,
        pos_label: Any | None = ...,
        prevalence_pos_label: Any | None = ...
    ) -> None: ...
    chance_level_: Any
    def plot(
        self,
        ax: Any | None = ...,
        *,
        name: Any | None = ...,
        plot_chance_level: bool = ...,
        chance_level_kw: Any | None = ...,
        **kwargs
    ): ...
    @classmethod
    def from_estimator(
        cls,
        estimator,
        X,
        y,
        *,
        sample_weight: Any | None = ...,
        pos_label: Any | None = ...,
        drop_intermediate: bool = ...,
        response_method: str = ...,
        name: Any | None = ...,
        ax: Any | None = ...,
        plot_chance_level: bool = ...,
        chance_level_kw: Any | None = ...,
        **kwargs
    ): ...
    @classmethod
    def from_predictions(
        cls,
        y_true,
        y_pred,
        *,
        sample_weight: Any | None = ...,
        pos_label: Any | None = ...,
        drop_intermediate: bool = ...,
        name: Any | None = ...,
        ax: Any | None = ...,
        plot_chance_level: bool = ...,
        chance_level_kw: Any | None = ...,
        **kwargs
    ): ...
