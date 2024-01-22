from typing import Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
from matplotlib.axes import Axes
from scipy.stats import gaussian_kde, ttest_ind

from predictables.univariate.src.plots.util import plot_label
from predictables.util import get_column_dtype, to_pd_s


def cdf_plot_matplotlib(
    x: Union[pl.Series, pd.Series, np.ndarray],
    plot_by: Union[pl.Series, pd.Series, np.ndarray],
    cv_folds: Union[pl.Series, pd.Series, np.ndarray],
    x_label: str = None,
    y_label: str = "Empirical Cumulative Distribution Function",
    ax: Axes = None,
    **kwargs,
) -> Axes:
    x = to_pd_s(x)
    plot_by = to_pd_s(plot_by)

    if ax is None:
        fig, ax = plt.subplots()

    # Plot the total CDFs
    ax = cdf_plot_matplotlib_levels(
        x=x,
        plot_by=plot_by,
        x_label=x_label,
        y_label=y_label,
        ax=ax,
        **kwargs,
    )

    # Plot the CV fold CDFs (lower alpha) to show the distribution
    ax = cdf_plot_matplotlib_levels_cv(
        x=x,
        plot_by=plot_by,
        cv_folds=cv_folds,
        x_label=x_label,
        y_label=y_label,
        ax=ax,
        alpha=0.3,
        **kwargs,
    )

    if x_label is not None:
        ax.set_xlabel(x_label)
    if y_label is not None:
        ax.set_ylabel(y_label)

    js_divergence_test = jenson_shannon_divergence()

    js_divergence_annotation = js_divergence_annotation()

    title = create_title()
    ax.set_title(title)

    plt.legend()

    return ax


def cdf_plot_matplotlib_levels(
    x: Union[pl.Series, pd.Series, np.ndarray],
    plot_by: Union[pl.Series, pd.Series, np.ndarray],
    x_label: str = None,
    y_label: str = "Empirical Cumulative Distribution Function",
    ax: Axes = None,
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

    Returns
    -------
    Axes
        The axes on which the CDF was plotted.
    """
    x = to_pd_s(x)
    plot_by = to_pd_s(plot_by)

    if ax is None:
        fig, ax = plt.subplots()

    # For each level of plot_by, plot the cdf of x, conditional on plot_by
    for level in plot_by.drop_duplicates().sort_values().values:
        label = plot_label(plot_by.name)
        label += f" = {level}"
        x_cdf = calculate_cdf(x[plot_by == level])
        ax = x_cdf.plot.line(ax=ax, label=label, **kwargs)

    if x_label is not None:
        ax.set_xlabel(x_label)
    if y_label is not None:
        ax.set_ylabel(y_label)

    return ax


def cdf_plot_matplotlib_levels_cv(
    x: Union[pl.Series, pd.Series, np.ndarray],
    plot_by: Union[pl.Series, pd.Series, np.ndarray],
    cv_folds: Union[pl.Series, pd.Series, np.ndarray],
    x_label: str = None,
    y_label: str = "Empirical Cumulative Distribution Function",
    ax: Axes = None,
    **kwargs,
) -> Axes:
    """
    Plots the empirical CDF of the given data for each level in `plot_by` and each fold in `cv_folds`. This plot is meant to show a "distribution" of possible CDF plots that could have been pulled from the real distribution.

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
        fig, ax = plt.subplots()

    # For each level of plot_by, plot the cdf of x, conditional on plot_by
    for level in plot_by.drop_duplicates().sort_values().values:
        for fold in cv_folds.drop_duplicates().sort_values().values:
            x_cdf = calculate_cdf(x[(plot_by == level) & (cv_folds == fold)])
            ax = x_cdf.plot.line(ax=ax, **kwargs)

    if x_label is not None:
        ax.set_xlabel(x_label)
    if y_label is not None:
        ax.set_ylabel(y_label)

    return ax


def create_title():
    raise NotImplementedError("Not implemented yet")
    return "title"


def jenson_shannon_divergence():
    raise NotImplementedError("Not implemented yet")
    return "js_divergence"


def js_divergence_annotation():
    raise NotImplementedError("Not implemented yet")
    return "js_divergence_annotation"


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
    x = x.sort_values()
    n = len(x)

    return pd.Series(np.arange(1, n + 1) / n, index=x)
