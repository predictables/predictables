from __future__ import annotations

from typing import Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
from matplotlib.axes import Axes
from scipy.spatial.distance import jensenshannon as js_divergence

from predictables.univariate.src.plots.util import binary_color, plot_label
from predictables.util import filter_by_cv_fold, get_column_dtype, to_pd_s


def cdf_plot(
    x: Union[pl.Series, pd.Series, np.ndarray],
    plot_by: Union[pl.Series, pd.Series, np.ndarray],
    cv_folds: Union[pl.Series, pd.Series, np.ndarray],
    x_label: Optional[str] = None,
    y_label: str = "Empirical Cumulative Distribution Function",
    ax: Optional[Axes] = None,
    figsize: Tuple[int, int] = (7, 7),
    backend: str = "matplotlib",
    time_series_validation: bool = True,
    **kwargs,
) -> Axes:
    """
    Plots the empirical CDF of the given data for each level in `plot_by` and each fold
    in `cv_folds`. This plot is meant to show a "distribution" of possible CDF plots
    that could have been pulled from the real distribution.

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
    time_series_validation : bool, optional
        Whether the cross-validation is based on a time series, by default True.

    Returns
    -------
    Axes
        The axes on which the CDF was plotted.
    """
    if backend == "matplotlib":
        if ax is None:
            _, ax_ = plt.subplots(figsize=figsize)
        else:
            ax_ = ax

        return cdf_plot_matplotlib(
            x=to_pd_s(x),
            plot_by=to_pd_s(plot_by),
            cv_folds=cv_folds,
            x_label=x_label,
            y_label=y_label,
            ax=ax_,
            time_series_validation=time_series_validation,
            **kwargs,
        )
    else:
        raise ValueError(f"Backend {backend} is not supported.")


def cdf_plot_matplotlib(
    x: Union[pl.Series, pd.Series, np.ndarray],
    plot_by: Union[pl.Series, pd.Series, np.ndarray],
    cv_folds: Union[pl.Series, pd.Series, np.ndarray],
    x_label: Union[str, None] = None,
    y_label: Union[str, None] = "Empirical Cumulative Distribution Function",
    ax: Union[Axes, None] = None,
    time_series_validation: bool = True,
    **kwargs,
) -> Axes:
    """
    Plots the empirical CDF of the given data for each level in `plot_by` and each
    fold in `cv_folds`. This plot is meant to show a "distribution" of possible CDF
    plots that could have been pulled from the real distribution. This plot is
    rendered using matplotlib.

    Whether using standard or time series cross-validation, the CDFs are plotted
    for the test set, to show the distribution of possible CDFs that could have
    been pulled from the real distribution and minimize overlap between the
    curves.
    """
    x = to_pd_s(x)
    plot_by = to_pd_s(plot_by)

    if ax is None:
        _, ax0 = plt.subplots()
    else:
        ax0 = ax

    _name = x.name if x.name is not None else "x"

    # Plot the total CDFs
    ax0 = cdf_plot_matplotlib_levels(
        x=x,
        plot_by=plot_by,
        x_label=(plot_label(x_label) if x_label is not None else plot_label(_name)),
        y_label=(
            y_label
            if y_label is not None
            else "Empirical Cumulative Distribution Function"
        ),
        ax=ax0,
        **kwargs,
    )

    # Plot the CV fold CDFs (lower alpha) to show the distribution
    ax0 = cdf_plot_matplotlib_levels_cv(
        x=x,
        plot_by=plot_by,
        cv_folds=cv_folds,
        x_label=(plot_label(x_label) if x_label is not None else plot_label(_name)),
        y_label=(
            y_label
            if y_label is not None
            else "Empirical Cumulative Distribution Function"
        ),
        ax=ax0,
        alpha=0.3,
        time_series_validation=time_series_validation,
        **kwargs,
    )

    if x_label is not None:
        ax0.set_xlabel(plot_label(x_label))
    else:
        ax0.set_xlabel(plot_label(_name))
    if y_label is not None:
        ax0.set_ylabel(y_label)
    else:
        ax0.set_ylabel("Empirical Cumulative Distribution Function")

    # Add horizontal line at 0 and 1
    ax0.axhline(0, color="black", linestyle="--", linewidth=1, label="_nolegend_")
    ax0.axhline(1, color="black", linestyle="--", linewidth=1, label="_nolegend_")

    title = create_title(_name, plot_by.name if plot_by.name is not None else "plot_by")
    ax0.set_title(title)

    plt.legend(
        loc="lower right", bbox_to_anchor=(1 - 0.01, 0 + 0.0575), borderaxespad=0.0
    )

    # show gridlines
    ax0.grid(True)

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
        _, ax_ = plt.subplots(figsize=figsize)
    else:
        ax_ = ax

    _plotby_name = plot_by.name if plot_by.name is not None else "plot_by"

    # For each level of plot_by, plot the cdf of x, conditional on plot_by
    for level in plot_by.drop_duplicates().sort_values().to_numpy():
        label = plot_label(_plotby_name)
        label += f" = {level}"
        x_cdf = calculate_cdf(x[plot_by == level])
        ax_ = x_cdf.plot.line(ax=ax_, label=label, color=binary_color(level), **kwargs)

    if x_label is not None:
        ax_.set_xlabel(x_label)
    if y_label is not None:
        ax_.set_ylabel(y_label)

    return ax_


def cdf_plot_matplotlib_levels_cv(
    x: Union[pl.Series, pd.Series, np.ndarray],
    plot_by: Union[pl.Series, pd.Series, np.ndarray],
    cv_folds: Union[pl.Series, pd.Series, np.ndarray],
    x_label: Optional[str] = None,
    y_label: Optional[str] = "Empirical Cumulative Distribution Function",
    ax: Optional[Axes] = None,
    time_series_validation: bool = True,
    **kwargs,
) -> Axes:
    """
    Plots the empirical CDF of the given data for each level in `plot_by`
    and each fold in `cv_folds`. This plot is meant to show a "distribution"
    of possible CDF plots that could have been pulled from the real
    distribution.

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
    time_series_validation : bool, optional
        Whether the cross-validation is based on a time series, by default True.

    Returns
    -------
    Axes
        The axes on which the CDF was plotted.
    """
    x = to_pd_s(x) if isinstance(x, pl.Series) else x
    plot_by = to_pd_s(plot_by) if plot_by is not None else None
    cv_folds = to_pd_s(cv_folds) if cv_folds is not None else None

    if ax is None:
        _, ax_ = plt.subplots()
    else:
        ax_ = ax

    # For each level of plot_by, plot the cdf of x, conditional on plot_by
    for level in plot_by.drop_duplicates().sort_values().to_numpy():
        if cv_folds is not None:
            for fold in cv_folds.drop_duplicates().sort_values().to_numpy():
                x_cdf = calculate_cdf(
                    filter_by_cv_fold(
                        x, fold, cv_folds, time_series_validation, "test"
                    )[(to_pd_s(plot_by) == level)]
                )
                ax_ = x_cdf.plot(
                    kind="line", ax=ax_, color=binary_color(level), **kwargs
                )
        else:
            return ax_

    (
        ax_.set_xlabel(x_label)
        if x_label is not None
        else ax_.set_xlabel(x.name if x.name is not None else "x")
    )
    (
        ax_.set_ylabel(y_label)
        if y_label is not None
        else ax_.set_ylabel("Empirical Cumulative Distribution Function")
    )
    return ax_


def jenson_shannon_divergence(
    x: Union[pl.Series, pd.Series, np.ndarray],
    plot_by: Union[pl.Series, pd.Series, np.ndarray],
) -> float:
    """
    Calculates the Jenson-Shannon divergence between `x` conditional on each
    level of `plot_by`.

    Parameters
    ----------
    x : Union[pl.Series, pd.Series, np.ndarray]
        The data to calculate the Jenson-Shannon divergence from.
    plot_by : Union[pl.Series, pd.Series, np.ndarray]
        The levels used to split the `x` data, and which form the different
        compared distributions.

    Returns
    -------
    float
        The Jenson-Shannon divergence between `x` conditioned on each level
        of `plot_by`.
    """
    pb_test = (
        plot_by.unique().sort()
        if isinstance(plot_by, pl.Series)
        else np.unique(plot_by)
    )
    x0 = x[plot_by == pb_test[0]]
    x1 = x[plot_by == pb_test[1]]
    return js_divergence(x0, x1)


def create_title(feature_name: str, target_name: str):
    return (
        f"Empirical CDF of {plot_label(feature_name)} conditioned on "
        f"{plot_label(target_name)}"
    )


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


def calculate_cdf(x: Union[pl.Series, pd.Series, np.ndarray]) -> pd.Series:
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
        if np.isinf(x.to_numpy()).any():
            raise ValueError("The array must not contain infs.")
    # Can't contain non-numeric values:
    if not np.issubdtype(x, np.number):
        raise ValueError("The array must not contain non-numeric values.")
    # Calculate the CDF
    x = x.sort_values()
    n = len(x)

    return pd.Series(np.arange(1, n + 1) / n, index=x)
