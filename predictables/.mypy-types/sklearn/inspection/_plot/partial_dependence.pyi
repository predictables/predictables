from .. import partial_dependence as partial_dependence
from ...base import is_regressor as is_regressor
from ...utils import (
    Bunch as Bunch,
    check_array as check_array,
    check_matplotlib_support as check_matplotlib_support,
    check_random_state as check_random_state,
)
from ...utils.parallel import Parallel as Parallel, delayed as delayed
from typing import Any

class PartialDependenceDisplay:
    pd_results: Any
    features: Any
    feature_names: Any
    target_idx: Any
    deciles: Any
    kind: Any
    subsample: Any
    random_state: Any
    is_categorical: Any
    def __init__(
        self,
        pd_results,
        *,
        features,
        feature_names,
        target_idx,
        deciles,
        kind: str = ...,
        subsample: int = ...,
        random_state: Any | None = ...,
        is_categorical: Any | None = ...
    ) -> None: ...
    @classmethod
    def from_estimator(
        cls,
        estimator,
        X,
        features,
        *,
        sample_weight: Any | None = ...,
        categorical_features: Any | None = ...,
        feature_names: Any | None = ...,
        target: Any | None = ...,
        response_method: str = ...,
        n_cols: int = ...,
        grid_resolution: int = ...,
        percentiles=...,
        method: str = ...,
        n_jobs: Any | None = ...,
        verbose: int = ...,
        line_kw: Any | None = ...,
        ice_lines_kw: Any | None = ...,
        pd_line_kw: Any | None = ...,
        contour_kw: Any | None = ...,
        ax: Any | None = ...,
        kind: str = ...,
        centered: bool = ...,
        subsample: int = ...,
        random_state: Any | None = ...
    ): ...
    bounding_ax_: Any
    figure_: Any
    axes_: Any
    lines_: Any
    contours_: Any
    bars_: Any
    heatmaps_: Any
    deciles_vlines_: Any
    deciles_hlines_: Any
    def plot(
        self,
        *,
        ax: Any | None = ...,
        n_cols: int = ...,
        line_kw: Any | None = ...,
        ice_lines_kw: Any | None = ...,
        pd_line_kw: Any | None = ...,
        contour_kw: Any | None = ...,
        bar_kw: Any | None = ...,
        heatmap_kw: Any | None = ...,
        pdp_lim: Any | None = ...,
        centered: bool = ...
    ): ...
