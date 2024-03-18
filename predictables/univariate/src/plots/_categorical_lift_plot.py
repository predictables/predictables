from __future__ import annotations

from typing import Tuple, Union

import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
from matplotlib.axes import Axes

from predictables.univariate.src.plots.util import plot_label
from predictables.util import get_column_dtype


def categorical_lift_plot(
    feature: pd.Series,
    target: pd.Series,
    feature_name: str,
    target_name: str,
    backend: str = "matplotlib",
    ax: Union[Axes, None] = None,
    figsize: Tuple[int, int] = (7, 7),
    **kwargs,
) -> Union[Axes, go.Figure, None]:
    """
    Plots the lift chart for a given categorical feature and target.

    Parameters
    ----------
    feature : pd.Series
        The categorical feature data.
    target : pd.Series
        The binary target data.
    feature_name : str
        The name of the feature.
    target_name : str
        The name of the target.
    backend : str, optional
        The plotting backend to use. Either "matplotlib" or "plotly".
    ax : matplotlib axes object, optional
        The axes on which to plot the lift chart. If None, a new figure and axes
        will be created.
    figsize : Tuple[int, int], optional
        The size of the figure to create if ax is None.
    **kwargs
        Additional keyword arguments to pass to the plotting function.

    Returns
    -------
    Union[Axes, go.Figure, None]
        The axes or figure containing the lift chart.
    """
    if backend == "matplotlib":
        return plot_cat_lift_plot(
            feature=feature,
            target=target,
            feature_name=feature_name,
            target_name=target_name,
            ax=ax,
            figsize=figsize,
            **kwargs,
        )
    elif backend == "plotly":
        return plotly_cat_lift_plot(
            feature=feature,
            target=target,
            feature_name=feature_name,
            target_name=target_name,
            **kwargs,
        )
    else:
        raise ValueError("backend must be either matplotlib or plotly")


def plot_cat_lift_plot(
    feature: pd.Series,
    target: pd.Series,
    feature_name: str,
    target_name: str,
    ax: Union[Axes, None] = None,
    figsize: Tuple[int, int] = (7, 7),
) -> Union[Axes, None]:
    """
    Plots the lift chart for a given categorical feature and target. Returns a
    matplotlib axes object.

    Parameters
    ----------
    ax : matplotlib axes object, optional
        The axes on which to plot the lift chart. If None, a new figure and axes
        will be created.

    Returns
    -------
    ax : matplotlib axes object
        The axes on which the lift chart was plotted.
    """
    feature_name = plot_label(feature_name)
    target_name = plot_label(target_name)

    lift_data = calculate_lift(feature=feature, target=target)

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    if ax is None:
        return None
    colors = ["green" if lift > 1 else "red" for lift in lift_data["lift"]]

    ax.bar(lift_data["Feature"], lift_data["lift"], color=colors, alpha=0.6)
    ax.axhline(1, color="black", linestyle="--")
    ax.set_xlabel(feature_name)
    ax.set_ylabel("Lift")
    ax.set_title(f"Lift Plot - Model Including {feature_name} vs Null Model")

    plt.tight_layout()

    # Add data label annotations
    for i, lift in enumerate(lift_data["lift"]):
        if lift != 0:
            ax.annotate(
                f"{lift:.2f}",
                xy=(i, lift),
                xytext=(0, -10),
                textcoords="offset points",
                ha="center",
                va="bottom",
                font={"size": 14 * (figsize[0] / 8)},
                bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9},
            )
        else:
            ax.annotate(
                f"{0:.2f}",
                xy=(i, 0),
                xytext=(0, 5),
                textcoords="offset points",
                ha="center",
                va="bottom",
                font={"size": 14 * (figsize[0] / 8)},
                bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9},
            )
    return ax


def plotly_cat_lift_plot(
    feature: pd.Series,
    target: pd.Series,
    feature_name: str,
    target_name: str,
    alpha: float = 0.5,
) -> go.Figure:
    """
    Plots the lift chart for a given categorical feature and target using Plotly.

    Parameters
    ----------
    feature : pd.Series
        The categorical feature data.
    target : pd.Series
        The binary target data.
    feature_name : str
        The name of the feature.
    target_name : str
        The name of the target.
    alpha : float, optional
        The opacity of the bars in the plot.

    Returns
    -------
    go.Figure
        The Plotly figure containing the lift chart.
    """
    feature_name = plot_label(feature_name)
    target_name = plot_label(target_name)
    lift_data = calculate_lift(feature, target)
    lift_data.columns = [feature_name, target_name, "lift"]

    colors = ["green" if lift > 1 else "red" for lift in lift_data["lift"]]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=lift_data[feature_name],
            y=lift_data["lift"],
            marker_color=colors,
            opacity=alpha,
        )
    )
    fig.add_hline(
        y=1, line={"color": "black", "dash": "dash"}, annotation_text="Baseline"
    )

    annotations = [
        {
            "x": row[feature_name],
            "y": row["lift"],
            "text": f"{row['lift']:.2f}",
            "font": {"family": "Arial", "size": 16, "color": "black"},
            "showarrow": False,
            "bgcolor": "white",
            "bordercolor": "black",
            "borderwidth": 1,
            "borderpad": 4,
        }
        for _, row in lift_data.iterrows()
    ]
    # Update layout
    fig.update_layout(
        title_text=f"Lift Plot - Model Including {feature_name} vs Null Model",
        xaxis_title=feature_name,
        yaxis_title="Lift",
        showlegend=False,
        annotations=annotations,
    )

    return fig


def calculate_lift(feature: pd.Series, target: pd.Series) -> pd.DataFrame:
    """
    Calculates the lift for a given feature and target.

    Parameters
    ----------
    feature : pd.Series
        The feature data (categorical).
    target : pd.Series
        The target data (binary).

    Returns
    -------
    pd.DataFrame
        A dataframe containing the feature, target mean, and lift values.
    """
    # Check for empty inputs and valid data types
    if feature.empty or target.empty:
        raise ValueError("Feature and target series must not be empty.")
    if get_column_dtype(feature) != "categorical":
        raise TypeError("Feature must be a categorical series.")
    if get_column_dtype(target) != "binary":
        raise TypeError("Target must be a binary (boolean) series.")

    overall_positive_rate = target.mean()
    df = pd.DataFrame({"Feature": feature, "Target": target})
    lift_data = df.groupby("Feature", observed=True)["Target"].mean().reset_index()
    lift_data["lift"] = lift_data["Target"] / overall_positive_rate

    return lift_data.sort_values("lift", ascending=False).reset_index(drop=True)
