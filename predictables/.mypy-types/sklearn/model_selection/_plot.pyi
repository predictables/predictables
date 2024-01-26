from ..utils import check_matplotlib_support as check_matplotlib_support
from ._validation import (
    learning_curve as learning_curve,
    validation_curve as validation_curve,
)
from typing import Any

class _BaseCurveDisplay: ...

class LearningCurveDisplay(_BaseCurveDisplay):
    train_sizes: Any
    train_scores: Any
    test_scores: Any
    score_name: Any
    def __init__(
        self, *, train_sizes, train_scores, test_scores, score_name: Any | None = ...
    ) -> None: ...
    def plot(
        self,
        ax: Any | None = ...,
        *,
        negate_score: bool = ...,
        score_name: Any | None = ...,
        score_type: str = ...,
        log_scale: str = ...,
        std_display_style: str = ...,
        line_kw: Any | None = ...,
        fill_between_kw: Any | None = ...,
        errorbar_kw: Any | None = ...
    ): ...
    @classmethod
    def from_estimator(
        cls,
        estimator,
        X,
        y,
        *,
        groups: Any | None = ...,
        train_sizes=...,
        cv: Any | None = ...,
        scoring: Any | None = ...,
        exploit_incremental_learning: bool = ...,
        n_jobs: Any | None = ...,
        pre_dispatch: str = ...,
        verbose: int = ...,
        shuffle: bool = ...,
        random_state: Any | None = ...,
        error_score=...,
        fit_params: Any | None = ...,
        ax: Any | None = ...,
        negate_score: bool = ...,
        score_name: Any | None = ...,
        score_type: str = ...,
        log_scale: str = ...,
        std_display_style: str = ...,
        line_kw: Any | None = ...,
        fill_between_kw: Any | None = ...,
        errorbar_kw: Any | None = ...
    ): ...

class ValidationCurveDisplay(_BaseCurveDisplay):
    param_name: Any
    param_range: Any
    train_scores: Any
    test_scores: Any
    score_name: Any
    def __init__(
        self,
        *,
        param_name,
        param_range,
        train_scores,
        test_scores,
        score_name: Any | None = ...
    ) -> None: ...
    def plot(
        self,
        ax: Any | None = ...,
        *,
        negate_score: bool = ...,
        score_name: Any | None = ...,
        score_type: str = ...,
        std_display_style: str = ...,
        line_kw: Any | None = ...,
        fill_between_kw: Any | None = ...,
        errorbar_kw: Any | None = ...
    ): ...
    @classmethod
    def from_estimator(
        cls,
        estimator,
        X,
        y,
        *,
        param_name,
        param_range,
        groups: Any | None = ...,
        cv: Any | None = ...,
        scoring: Any | None = ...,
        n_jobs: Any | None = ...,
        pre_dispatch: str = ...,
        verbose: int = ...,
        error_score=...,
        fit_params: Any | None = ...,
        ax: Any | None = ...,
        negate_score: bool = ...,
        score_name: Any | None = ...,
        score_type: str = ...,
        std_display_style: str = ...,
        line_kw: Any | None = ...,
        fill_between_kw: Any | None = ...,
        errorbar_kw: Any | None = ...
    ): ...
