"""Provide plotting support for the Predictables project."""

from __future__ import annotations
import numpy as np
import pandas as pd
from bokeh.plotting import figure
from bokeh.models import Whisker, ColumnDataSource, HoverTool

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


def scatter(
    X: np.ndarray,
    y: np.ndarray,
    X_name: str,
    y_name: str,
    plot_title: str | None = None,
    plot_width: int = PLOT_WIDTH,
    plot_height: int = PLOT_HEIGHT,
) -> figure:
    """Create a scatter plot."""
    if plot_title is None:
        plot_title = f"Scatter plot of `{X_name}` by `{y_name}`"

    p = figure(
        title=plot_title,
        x_axis_label=X_name,
        y_axis_label=y_name,
        width=plot_width,
        height=plot_height,
    )

    p.circle(X, y, size=8, color="navy", alpha=0.5)

    return p


def _compute_quantiles(X: np.ndarray, y: np.ndarray) -> pd.DataFrame:
    """Compute quantiles for the given feature and target arrays."""
    data = pd.DataFrame({"X": X, "y": y})
    quantiles = (
        data.groupby("y")["X"].quantile([0.25, 0.5, 0.75]).unstack().reset_index()  # noqa: PD010
    )
    quantiles.columns = ["y", "q1", "q2", "q3"]
    return quantiles


def boxplot(
    X: np.ndarray,
    y: np.ndarray,
    X_name: str,
    y_name: str,
    plot_title: str | None = None,
    plot_width: int = PLOT_WIDTH,
    plot_height: int = PLOT_HEIGHT,
    box_width: float = 0.5,
) -> figure:
    """Create a box plot for X by the levels of y."""
    if plot_title is None:
        plot_title = f"Box plot of `{X_name}` by level of `{y_name}`"

    quantiles = _compute_quantiles(X, y)
    quantiles["iqr"] = quantiles["q3"] - quantiles["q1"]
    quantiles["upper"] = quantiles["q3"] + 1.5 * quantiles["iqr"]
    quantiles["lower"] = quantiles["q1"] - 1.5 * quantiles["iqr"]

    # Merge quantiles with the original data
    data = pd.DataFrame({"X": X, "y": y})
    data = data.merge(quantiles, on="y", how="left")

    # Indicate if the value is an outlier
    data["is_outlier"] = (data["X"] < data["lower"]) | (data["X"] > data["upper"])

    source = ColumnDataSource(data)

    p = figure(
        title=plot_title,
        x_axis_label=y_name,
        y_axis_label=X_name,
        width=plot_width,
        height=plot_height,
        tools="hover,box_select,lasso_select,reset,tap",
        tooltips=[("Value", "@X"), ("Level", "@y")],
    )

    # Outlier range
    whisker = Whisker(
        base="y",
        upper="upper",
        lower="lower",
        source=source,
        line_color="black",
        line_width=2,
        line_alpha=0.5,
    )
    p.add_layout(whisker)

    # Quantile boxes
    p.vbar(
        x="y",
        width=box_width,
        bottom="q1",
        top="q3",
        source=source,
        fill_color="skyblue",
        line_color="black",
        fill_alpha=0.6,
    )

    # Median line
    p.vbar(x="y", width=0.2, bottom="q2", top="q2", source=source, line_color="black")

    # # Outliers
    outlier_source = data.loc[data.is_outlier]
    outliers = p.scatter(
        x="y",
        y="X",
        source=outlier_source,
        size=12,
        selection_color="firebrick",
        selection_fill_alpha=0.5,
        nonselection_line_color="black",
        nonselection_fill_color="green",
        nonselection_fill_alpha=0.5,
    )

    p.hover.renderers = [outliers]
    return p
