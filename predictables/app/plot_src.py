"""Provide plotting support for the Predictables project."""

from __future__ import annotations
import numpy as np
import pandas as pd
from bokeh.plotting import figure
from bokeh.models import Whisker, ColumnDataSource
from bokeh.transform import factor_cmap

from .constants import (
    PLOT_WIDTH,
    PLOT_HEIGHT,
    N_HISTOGRAM_BINS,
    IS_HISTOGRAM_DENSITY,
    FILL_ALPHA,
    FILL_COLOR,
    LINE_COLOR,
)


def histogram(
    X: np.ndarray,
    X_name: str,
    plot_title: str = "Distribution of `X`",
    n_bins: int = N_HISTOGRAM_BINS,
    density: bool = IS_HISTOGRAM_DENSITY,
    plot_width: int = PLOT_WIDTH,
    plot_height: int = PLOT_HEIGHT,
    fill_alpha: float = FILL_ALPHA,
    fill_color: str = FILL_COLOR,
    line_color: str = LINE_COLOR,
) -> figure:
    """Create a histogram plot."""
    bins = np.linspace(X.min(), X.max(), n_bins)

    hist, edges = np.histogram(X, bins=bins, density=density)

    p = figure(
        title=plot_title,
        x_axis_label=X_name,
        y_axis_label="Count",
        width=plot_width,
        height=plot_height,
    )

    p.quad(
        top=hist,
        bottom=0,
        left=edges[:-1],
        right=edges[1:],
        fill_color=fill_color,
        line_color=line_color,
        fill_alpha=fill_alpha,
    )

    return p


def scatter():
    pass


def _compute_quantiles(X: np.ndarray, y: np.ndarray) -> pd.DataFrame:
    """Compute the quantiles for the box plot."""
    df = pd.DataFrame({"X": X, "y": y})
    quantiles = (
        df.groupby("y")["X"].quantile([0.25, 0.5, 0.75]).unstack(level=1).reset_index()  # noqa: PD010
    )
    quantiles.columns = ["y", "q1", "q2", "q3"]

    return df.merge(quantiles, on="y", how="left")


def boxplot(
    X: np.ndarray,
    y: np.ndarray,
    X_name: str,
    y_name: str,
    plot_title: str | None = None,
    plot_width: int = PLOT_WIDTH,
    plot_height: int = PLOT_HEIGHT,
) -> figure:
    """Create a box plot for X by the levels of y."""
    if plot_title is None:
        plot_title = f"Box plot of `{X_name}` by level of `{y_name}`"

    kinds = y.unique()

    quantiles = _compute_quantiles(X, y)
    quantiles["iqr"] = quantiles["q3"] - quantiles["q1"]

    quantiles["upper"] = quantiles["q3"] + 1.5 * quantiles["iqr"]
    quantiles["lower"] = quantiles["q1"] - 1.5 * quantiles["iqr"]

    quantiles["y"] = quantiles["y"].astype(str)

    source = ColumnDataSource(quantiles)

    p = figure(
        title=plot_title,
        x_axis_label=y_name,
        y_axis_label=X_name,
        width=plot_width,
        height=plot_height,
    )

    # Outlier range
    whiskers = {}
    for kind in kinds:
        whiskers[kind] = Whisker(
            base="y",
            upper="upper",
            lower="lower",
            line_color="black",
            line_width=2,
            line_alpha=0.5,
            source=source,
        )

        whiskers[kind].upper_head.size = 10
        p.add_layout(whiskers[kind])

    # Quantile boxes
    p.vbar(
        x="y",
        top="q3",
        bottom="q1",
        width=0.4,
        source=source,
        fill_color=factor_cmap("y", palette=["#FF5733", "#33FF57"], factors=kinds),
        line_color="black",
        fill_alpha=0.6,
    )

    # Add outliers
    outliers = quantiles[
        (quantiles["X"] > quantiles["upper"]) | (quantiles["X"] < quantiles["lower"])
    ]
    p.circle(x="y", y="X", source=outliers, color="black", size=6)

    return p
