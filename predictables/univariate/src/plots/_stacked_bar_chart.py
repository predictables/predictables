from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import polars as pl

from .util import plot_label, rotate_x_lab


def stacked_bar_chart(
    feature: [pl.Series, pd.Series],
    target: [pl.Series, pd.Series],
    backend: str = "matplotlib",
    y_offset: float = 0.035,
    ax: plt.Axes | None = None,
    figsize: tuple[float, float] = (7, 7),
    alpha: float = 0.8,
    bar_width: float = 0.8,
    fontsize: int = 16,
    facecolor: str = "white",
) -> plt.Axes | go.Figure:
    params = dict(
        feature=feature,
        target=target,
        y_offset=y_offset,
        ax=ax,
        figsize=figsize,
        alpha=alpha,
        bar_width=bar_width,
        fontsize=fontsize,
        facecolor=facecolor,
    )
    if backend == "matplotlib":
        return plot_stacked_bar_chart(**params, ax=ax, figsize=figsize)
    elif backend == "plotly":
        return plotly_stacked_bar_chart(**params, bar_width=bar_width)
    else:
        raise ValueError(f"Unknown backend: {backend}")


def plot_stacked_bar_chart(
    feature: [pl.Series, pd.Series],
    target: [pl.Series, pd.Series],
    y_offset: float = 0.035,
    ax=None,
    figsize: tuple[float, float] = (7, 7),
    alpha: float = 0.8,
    bar_width: float = 0.8,
    fontsize: int = 16,
    facecolor: str = "white",
) -> plt.Axes | go.Figure:
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    # Assume fitted is a DataFrame containing the feature and target
    ct = pd.crosstab(feature, target, normalize="index").sort_values(0)

    n = len(ct.index)
    bar_width = bar_width
    indices = np.arange(n)

    bottoms = np.zeros(n)

    for col in ct.columns:
        ax.bar(indices, ct[col], bottom=bottoms, label=col, alpha=alpha)
        bottoms += ct[col].values

    # get the feature/target names
    feature_name = feature if isinstance(feature, str) else feature.name
    target_name = target if isinstance(target, str) else target.name

    # For each bar in each container
    for bar in ax.patches:
        # Get the y position and height of the bar
        y = bar.get_y() + bar.get_height()

        # Get the remaining portion of the bar not covered by the annotation
        y2 = 1 - y

        # Get the width of the bar and find the center
        # then shift the x position by the width
        x_value = bar.get_x() + bar_width / 2

        if abs(y - 1) > 1e-6:  # Tolerance to account for float arithmetic
            ax.annotate(
                f"{y*100:.1f}%",
                xy=(x_value, y - y_offset),
                textcoords="data",
                va="center",
                ha="center",
                fontsize=fontsize,
                bbox=dict(
                    facecolor=facecolor,
                    edgecolor="black",
                    alpha=0.9,
                    boxstyle="round,pad=0.2",
                ),
            )

            ax.annotate(
                f"{y2*100:.1f}%",
                xy=(x_value, 1 - y_offset),
                textcoords="data",
                va="center",
                ha="center",
                fontsize=fontsize,
                bbox=dict(
                    facecolor=facecolor,
                    edgecolor="black",
                    alpha=0.9,
                    boxstyle="round,pad=0.2",
                ),
            )

    # Set x and y labels
    ax.set_ylabel("Count")

    # Set y ticks to percentage
    yticks = ax.get_yticks()
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"{y:.1%}" for y in yticks])

    # Set title
    ax.set_title(
        f"Distribution of [{plot_label(feature_name)}] by[{plot_label(target_name)}]"
    )

    ax.set_xticks(indices)
    ax.set_xticklabels(ct.index)

    ax.legend(fontsize=fontsize)

    ax = rotate_x_lab(ax)
    ax.figure.canvas.draw()
    ax.figure.tight_layout()

    return ax


def plotly_stacked_bar_chart(
    feature: pl.Series | pd.Series,
    target: pl.Series | pd.Series,
    bar_width: float = 0.8,
    fontsize: int = 16,
    facecolor: str = "white",
) -> go.Figure:
    # Assume fitted is a DataFrame containing the feature and target
    ct = pd.crosstab(feature, target, normalize="index").sort_values(by=0)

    # Create an empty figure
    fig = go.Figure()

    # Create a stacked bar chart
    for col in ct.columns:
        fig.add_trace(
            go.Bar(
                x=ct.index,
                y=ct[col],
                name=str(col),  # Convert column name to string in case it's not
                width=[bar_width] * len(ct.index),  # Specify the width for each bar
            )
        )

    # Add annotations
    bottoms = np.zeros(len(ct.index))
    for col in ct.columns:
        for i, idx in enumerate(ct.index):
            y_value = ct.loc[idx, col]
            if y_value > 0:
                fig.add_annotation(
                    x=idx,
                    y=y_value / 2 + bottoms[i],
                    text=f"{y_value:.1%}",
                    showarrow=False,
                    font=dict(size=fontsize),
                    bgcolor=facecolor,
                    opacity=0.8,
                )
            bottoms[i] += y_value

    # Update the layout
    fig.update_layout(
        barmode="stack",
        title=f"Distribution of {feature} by {target}",
        xaxis=dict(
            title=feature,
            tickmode="array",
            tickvals=np.arange(len(ct.index)),
            ticktext=ct.index,
        ),
        yaxis=dict(title="Percentage", tickformat=".1%"),
        legend=dict(font=dict(size=fontsize), title="Legend"),
        plot_bgcolor="rgba(0,0,0,0)",  # Set transparent background color
    )

    # Update xaxis tickangle if necessary
    fig.update_xaxes(tickangle=-45)

    # Return the figure object for use in Dash
    return fig
