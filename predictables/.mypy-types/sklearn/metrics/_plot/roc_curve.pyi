from ...utils._plotting import _BinaryClassifierCurveDisplayMixin
from .._ranking import auc as auc, roc_curve as roc_curve
from typing import Any

class RocCurveDisplay(_BinaryClassifierCurveDisplayMixin):
    estimator_name: Any
    fpr: Any
    tpr: Any
    roc_auc: Any
    pos_label: Any
    def __init__(
        self,
        *,
        fpr,
        tpr,
        roc_auc: Any | None = ...,
        estimator_name: Any | None = ...,
        pos_label: Any | None = ...
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
        drop_intermediate: bool = ...,
        response_method: str = ...,
        pos_label: Any | None = ...,
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
        drop_intermediate: bool = ...,
        pos_label: Any | None = ...,
        name: Any | None = ...,
        ax: Any | None = ...,
        plot_chance_level: bool = ...,
        chance_level_kw: Any | None = ...,
        **kwargs
    ): ...
