from ...utils._plotting import _BinaryClassifierCurveDisplayMixin
from .._ranking import det_curve as det_curve
from typing import Any

class DetCurveDisplay(_BinaryClassifierCurveDisplayMixin):
    fpr: Any
    fnr: Any
    estimator_name: Any
    pos_label: Any
    def __init__(
        self, *, fpr, fnr, estimator_name: Any | None = ..., pos_label: Any | None = ...
    ) -> None: ...
    @classmethod
    def from_estimator(
        cls,
        estimator,
        X,
        y,
        *,
        sample_weight: Any | None = ...,
        response_method: str = ...,
        pos_label: Any | None = ...,
        name: Any | None = ...,
        ax: Any | None = ...,
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
        name: Any | None = ...,
        ax: Any | None = ...,
        **kwargs
    ): ...
    def plot(self, ax: Any | None = ..., *, name: Any | None = ..., **kwargs): ...
