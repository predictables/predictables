from typing import Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
from matplotlib.axes import Axes
from scipy.spatial.distance import jensenshannon as js_divergence

from predictables.univariate.src.plots.util import binary_color, plot_label
from predictables.util import get_column_dtype, to_pd_s


def cdf_plot(
    x: Union[pl.Series, pd.Series, np.ndarray],
    plot_by: Union[pl.Series, pd.Series, np.ndarray],
    cv_folds: Union[pl.Series, pd.Series, np.ndarray],
    x_label: Union[str, None] = None,
    y_label: str = "Empirical Cumulative Distribution Function",
    ax: Union[Axes, None] = None,
    figsize: Tuple[int, int] = (7, 7),
    backend: str = "matplotlib",
    **kwargs,
) -> Axes:
    """
    Plots the empirical CDF of the given data for each level in `plot_by` and each fold in
    `cv_folds`. This plot is meant to show a "distribution" of possible CDF plots that
    could have been pulled from the real distribution.

    Parameters
    ----------
    x : Union[pl.Series, pd.Series, np.ndarray]
        The data to plot the CDF from.
    plot_by : Union[pl.Series, pd.Series, np.ndarray]
        The data to plot the CDF from.
    cv_folds : Union[pl.Series, pd.Series, np.ndarray]
        Cross-validation folds used to plot a distribution of CDFs
    x_label : str, optional
        The label of the x-axis.
    y_label : str
        The label of the y-axis.
    ax : Axes, optional
        The axes to plot the CDF on, by default None.
    figsize : Tuple[int, int], optional
        The size of the figure, by default (7, 7).
    backend : str, optional
        The plotting backend to use, by default "matplotlib".

    Returns
    -------
    Axes
        The axes on which the CDF was plotted.
    """
    x = to_pd_s(x)
    plot_by = to_pd_s(plot_by)

    if backend == "matplotlib":
        if ax is None:
            _, ax = plt.subplots(figsize=figsize)

        ax = cdf_plot_matplotlib(
            x=x,
            plot_by=plot_by,
            cv_folds=cv_folds,
            x_label=x_label,
            y_label=y_label,
            ax=ax,
            **kwargs,
        )
    return ax


def cdf_plot_matplotlib(
    x: Union[pl.Series, pd.Series, np.ndarray],
    plot_by: Union[pl.Series, pd.Series, np.ndarray],
    cv_folds: Union[pl.Series, pd.Series, np.ndarray],
    x_label: Union[str, None] = None,
    y_label: Union[str, None] = "Empirical Cumulative Distribution Function",
    ax: Union[Axes, None] = None,
    **kwargs,
) -> Axes:
    """
    Plots the empirical CDF of the given data for each level in `plot_by` and each fold in `cv_folds`. This plot is meant to show a "distribution" of possible CDF plots that could have been pulled from the real distribution. This plot is rendered using matplotlib.
    """
    x = to_pd_s(x)
    plot_by = to_pd_s(plot_by)

    if ax is None:
        _, ax0 = plt.subplots()
    else:
        ax0 = ax

    # Plot the total CDFs
    ax0 = cdf_plot_matplotlib_levels(
        x=x,
        plot_by=plot_by,
        x_label=plot_label(x_label) if x_label is not None else plot_label(x.name),
        y_label=y_label
        if y_label is not None
        else "Empirical Cumulative Distribution Function",
        ax=ax0,
        **kwargs,
    )

    # Plot the CV fold CDFs (lower alpha) to show the distribution
    ax0 = cdf_plot_matplotlib_levels_cv(
        x=x,
        plot_by=plot_by,
        cv_folds=cv_folds,
        x_label=plot_label(x_label) if x_label is not None else plot_label(x.name),
        y_label=y_label
        if y_label is not None
        else "Empirical Cumulative Distribution Function",
        ax=ax0,
        alpha=0.3,
        **kwargs,
    )

    if x_label is not None:
        ax0.set_xlabel(plot_label(x_label))
    else:
        ax0.set_xlabel(plot_label(x.name))
    if y_label is not None:
        ax0.set_ylabel(y_label)
    else:
        ax0.set_ylabel("Empirical Cumulative Distribution Function")

    # js_divergence_test = jenson_shannon_divergence()

    # js_divergence_annotation = js_divergence_annotation()

    # Add horizontal line at 0 and 1
    ax0.axhline(0, color="black", linestyle="--", linewidth=1, label="_nolegend_")
    ax0.axhline(1, color="black", linestyle="--", linewidth=1, label="_nolegend_")

    title = create_title(x.name, plot_by.name)
    ax0.set_title(title)

    plt.legend(
        loc="lower right", bbox_to_anchor=(1 - 0.01, 0 + 0.0575), borderaxespad=0.0
    )

    return ax0


def cdf_plot_matplotlib_levels(
    x: Union[pl.Series, pd.Series, np.ndarray],
    plot_by: Union[pl.Series, pd.Series, np.ndarray],
    x_label: Optional[Union[str, None]] = None,
    y_label: Optional[str] = "Empirical Cumulative Distribution Function",
    ax: Optional[Axes] = None,
    figsize: Optional[Tuple[int, int]] = (7, 7),
    **kwargs,
) -> Axes:
    """
    Plots the empirical CDF of the given data.

    Parameters
    ----------
    x : Union[pl.Series, pd.Series, np.ndarray]
        The data to plot the CDF from.
    plot_by : Union[pl.Series, pd.Series, np.ndarray]
        The data to plot the CDF from.
    x_label : str, optional
        The label of the x-axis.
    y_label : str
        The label of the y-axis.
    ax : Axes, optional
        The axes to plot the CDF on, by default None.
    figsize : Tuple[int, int], optional
        The size of the figure, by default (7, 7).

    Returns
    -------
    Axes
        The axes on which the CDF was plotted.
    """
    x = to_pd_s(x)
    plot_by = to_pd_s(plot_by)

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    # For each level of plot_by, plot the cdf of x, conditional on plot_by
    for level in plot_by.drop_duplicates().sort_values().values:
        label = plot_label(plot_by.name)
        label += f" = {level}"
        x_cdf = calculate_cdf(x[plot_by == level])
        ax = x_cdf.plot.line(ax=ax, label=label, color=binary_color(level), **kwargs)

    if x_label is not None:
        ax.set_xlabel(x_label)
    if y_label is not None:
        ax.set_ylabel(y_label)

    return ax


def cdf_plot_matplotlib_levels_cv(
    x: Union[pl.Series, pd.Series, np.ndarray],
    plot_by: Union[pl.Series, pd.Series, np.ndarray],
    cv_folds: Union[pl.Series, pd.Series, np.ndarray],
    x_label: Optional[Union[str, None]] = None,
    y_label: Optional[str] = "Empirical Cumulative Distribution Function",
    ax: Optional[Union[Axes, None]] = None,
    **kwargs,
) -> Axes:
    """
    Plots the empirical CDF of the given data for each level in `plot_by` and each fold in
    `cv_folds`. This plot is meant to show a "distribution" of possible CDF plots that could
    have been pulled from the real distribution.

    Parameters
    ----------
    x : Union[pl.Series, pd.Series, np.ndarray]
        The data to plot the CDF from.
    plot_by : Union[pl.Series, pd.Series, np.ndarray]
        The data to plot the CDF from.
    cv_folds : Union[pl.Series, pd.Series, np.ndarray]
        Cross-validation folds used to plot a distribution of CDFs
    x_label : str, optional
        The label of the x-axis.
    y_label : str
        The label of the y-axis.
    ax : Axes, optional
        The axes to plot the CDF on, by default None.

    Returns
    -------
    Axes
        The axes on which the CDF was plotted.
    """
    x = to_pd_s(x)
    plot_by = to_pd_s(plot_by)

    if ax is None:
        _, ax = plt.subplots()

    # For each level of plot_by, plot the cdf of x, conditional on plot_by
    for level in plot_by.drop_duplicates().sort_values().values:
        for fold in cv_folds.drop_duplicates().sort_values().values:
            x_cdf = calculate_cdf(x[(plot_by == level) & (cv_folds == fold)])
            ax = x_cdf.plot.line(ax=ax, color=binary_color(level), **kwargs)

    ax.set_xlabel(x_label) if x_label is not None else ax.set_xlabel(x.name)
    ax.set_ylabel(y_label) if y_label is not None else ax.set_ylabel(
        "Empirical Cumulative Distribution Function"
    )
    return ax


def jenson_shannon_divergence(
    x: Union[pl.Series, pd.Series, np.ndarray],
    plot_by: Union[pl.Series, pd.Series, np.ndarray],
) -> float:
    """
    Calculates the Jenson-Shannon divergence between `x` conditional on each level of `plot_by`.

    Parameters
    ----------
    x : Union[pl.Series, pd.Series, np.ndarray]
        The data to calculate the Jenson-Shannon divergence from.
    plot_by : Union[pl.Series, pd.Series, np.ndarray]
        The levels used to split the `x` data, and which form the different compared distributions.

    Returns
    -------
    float
        The Jenson-Shannon divergence between `x` conditioned on each level of `plot_by`.
    """
    x0 = x[plot_by == plot_by.drop_duplicates().sort_values()[0]]
    x1 = x[plot_by == plot_by.drop_duplicates().sort_values()[1]]
    return js_divergence(x0, x1)


def create_title(feature_name: str, target_name: str):
    return f"CDF of {plot_label(feature_name)} conditioned on {plot_label(target_name)}"


def js_divergence_annotation(jsd: float) -> str:
    if jsd < 0.1:
        return (
            f"JS divergence of {jsd} indicates that the distributions are very similar"
        )
    elif jsd < 0.3:
        return "The distributions are somewhat similar"
    elif jsd < 0.5:
        return "The distributions are somewhat different"
    elif jsd < 0.7:
        return "The distributions are very different"
    else:
        return "The distributions are extremely different"


def calculate_cdf(
    x: Union[pl.Series, pd.Series, np.ndarray],
) -> pd.Series:
    """
    Calculates the empirical CDF from the given data.

    Parameters
    ----------
    x : Union[pl.Series, pd.Series, np.ndarray]
        The data to calculate the CDF from.

    Returns
    -------
    pd.Series
        The empirical CDF.
    """
    x = to_pd_s(x)

    # Validate inputs
    # Can't be empty:
    if x.shape[0] == 0:
        raise ValueError("The array must not be empty.")
    # Can't contain NaNs:
    if x.isna().any():
        raise ValueError("The array must not contain NaNs.")
    # Can't contain infs:
    if get_column_dtype(x) == "continuous":
        if np.isinf(x.values).any():
            raise ValueError("The array must not contain infs.")
    # Can't contain non-numeric values:
    if not np.issubdtype(x, np.number):
        raise ValueError("The array must not contain non-numeric values.")
    # Calculate the CDF
    x = x.sort_values()
    n = len(x)

    return pd.Series(np.arange(1, n + 1) / n, index=x)
