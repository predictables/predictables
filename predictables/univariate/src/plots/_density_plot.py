from __future__ import annotations

from functools import wraps
from typing import Callable, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
from matplotlib.axes import Axes
from scipy.stats import gaussian_kde, ttest_ind  # type: ignore

from predictables.univariate.src.plots.util import binary_color, plot_label
from predictables.util import (
    filter_by_cv_fold,
    get_column_dtype,
    graph_min_max,
    to_pd_s,
)


def density_plot(
    x: Union[pd.Series, pl.Series],
    plot_by: Union[pd.Series, pl.Series],
    cv_label: Union[pd.Series, pl.Series],
    x_min: Optional[float] = None,
    x_max: Optional[float] = None,
    ax: Optional[Axes] = None,
    grid_bins: int = 200,
    cv_alpha: float = 0.5,
    cv_line_width: float = 0.5,
    t_test_alpha: float = 0.05,
    figsize: Tuple[int, int] = (7, 7),
    call_legend: bool = True,
    backend: str = "matplotlib",
    time_series_validation: bool = True,
) -> Axes:
    """
    Plot density function as well as cross-validation densities using the
    specified backend.

    Parameters
    ----------
    x : Union[pd.Series, pl.Series]
        The variable to plot the density of.
    plot_by : Union[pd.Series, pl.Series]
        The variable to group by. For a binary target, this is the target.
        The plot will generate a density for each level of the target.
    cv_label : Union[pd.Series, pl.Series]
        The cross-validation fold to group by. If None, no grouping is done.
        Defaults to None.
    x_min : float, optional
        The minimum value to plot. If None, defaults to the minimum of x
        before grouping by any variables. Used to extend the curve to the
        edges of the plot.
    x_max : float, optional
        The maximum value to plot. If None, defaults to the maximum of x
        before grouping by any variables. Used to extend the curve to the
        edges of the plot.
    ax : matplotlib.axes.Axes, optional
        The axes to plot on. If None, a new figure and axes is created.
    grid_bins : int, optional
        The number of bins to use for the density. Defaults to 200.
    cv_alpha : float, optional
        The alpha value to use for the cross-validation folds. Defaults to
        0.5.
    cv_line_width : float, optional
        The width of the line to use for the cross-validation folds.
        Defaults to 0.5.
    t_test_alpha : float, optional
        The alpha value to use for the t-test. Defaults to 0.05.
    figsize : tuple, optional
        The size of the figure to create. Defaults to (7, 7). Only used
        if ax is None.
    call_legend : bool, optional
        Whether to call plt.legend() at the end of the function.
        Defaults to True.
    backend : str, optional
        The backend to use for plotting. Defaults to "matplotlib".
    time_series_validation : bool, optional
        Whether the cross-validation is based on a time series. Defaults
        to True.

    Returns
    -------
    matplotlib.axes.Axes
        The axes the plot was drawn on.

    """
    if backend == "matplotlib":
        return density_plot_mpl(
            x,
            plot_by,
            cv_label,
            x_min=x_min,
            x_max=x_max,
            ax=ax,
            grid_bins=grid_bins,
            cv_alpha=cv_alpha,
            cv_line_width=cv_line_width,
            t_test_alpha=t_test_alpha,
            figsize=figsize,
            call_legend=call_legend,
            time_series_validation=time_series_validation,
        )
    elif backend == "plotly":
        raise NotImplementedError(
            "Plotly backend not yet implemented. Use 'matplotlib' for now."
        )
    else:
        raise ValueError(f"Invalid backend {backend}.")


def validate_density_plot_mpl(func: Callable):
    """
    Decorator to validate the inputs to the density plot functions.

    Parameters
    ----------
    func : Callable
        The function to decorate.

    Returns
    -------
    Callable
        The decorated function.

    Raises
    ------
    ValueError
        If any of the inputs are invalid.
    """

    @wraps(func)
    def wrapper(
        x: Union[pd.Series, pl.Series],
        plot_by: Union[pd.Series, pl.Series],
        cv_label: Union[pd.Series, pl.Series],
        x_min: Optional[float] = None,
        x_max: Optional[float] = None,
        ax: Optional["plt.axes.Axes"] = None,
        grid_bins: int = 200,
        cv_alpha: float = 0.5,
        cv_line_width: float = 0.5,
        t_test_alpha: float = 0.05,
        figsize: Tuple[int, int] = (7, 7),
        call_legend: bool = True,
        time_series_validation: bool = True,
    ):
        # Input validations
        if not isinstance(x, (pd.Series, pl.Series)):
            raise ValueError(
                f"`x` must be a pandas or polars Series, but got {type(x)}."
            )

        if not isinstance(plot_by, (pd.Series, pl.Series)):
            raise ValueError(
                f"`plot_by` must be a pandas or polars Series, but got {type(plot_by)}."
            )

        if not isinstance(cv_label, (pd.Series, pl.Series)):
            raise ValueError(
                f"`cv_label` must be a pandas or polars Series, but got {type(cv_label)}."
            )

        def validate_float_int(value, arg_name):
            try:
                # Attempt to convert the value to a float
                return float(value)
            except ValueError:
                # If conversion fails, raise an informative error
                raise ValueError(
                    f"Argument `{arg_name}` must be convertible to float, but got value `{value}` of type `{type(value).__name__}`."
                )

        # Inside the wrapper function of the decorator
        if x_min is not None:
            x_min = validate_float_int(x_min, "x_min")

        if x_max is not None:
            x_max = validate_float_int(x_max, "x_max")

        if not isinstance(grid_bins, int):
            raise ValueError(f"`grid_bins` must be an int, but got {type(grid_bins)}.")

        if not isinstance(figsize, tuple) or len(figsize) != 2:
            raise ValueError(
                f"`figsize` must be a tuple of length 2, but got {type(figsize)} with length {len(figsize)}."
            )

        # Call the decorated function if all validations pass
        return func(
            x,
            plot_by,
            cv_label,
            x_min,
            x_max,
            ax,
            grid_bins,
            cv_alpha,
            cv_line_width,
            t_test_alpha,
            figsize,
            call_legend,
            time_series_validation,
        )

    return wrapper


@validate_density_plot_mpl
def density_plot_mpl(
    x: Union[pd.Series, pl.Series],
    plot_by: Union[pd.Series, pl.Series],
    cv_label: Union[pd.Series, pl.Series],
    x_min: Optional[float] = None,
    x_max: Optional[float] = None,
    ax: Optional[Axes] = None,
    grid_bins: int = 200,
    cv_alpha: float = 0.5,
    cv_line_width: float = 0.5,
    t_test_alpha: float = 0.05,
    figsize: Tuple[int, int] = (7, 7),
    call_legend: bool = True,
    time_series_validation: bool = True,
) -> Axes:
    """
    Plot density function as well as cross-validation densities.

    Parameters
    ----------
    x : Union[pd.Series, pl.Series]
        The variable to plot the density of.
    plot_by : Union[pd.Series, pl.Series]
        The variable to group by. For a binary target, this is the target.
        The plot will generate a density for each level of the target.
    cv_label : Union[pd.Series, pl.Series]
        The cross-validation fold to group by. If None, no grouping is
        done. Defaults to None.
    x_min : float, optional
        The minimum value to plot. If None, defaults to the minimum of
        x before grouping by any variables. Used to extend the curve to
        the edges of the plot.
    x_max : float, optional
        The maximum value to plot. If None, defaults to the maximum of
        x before grouping by any variables. Used to extend the curve to
        the edges of the plot.
    ax : matplotlib.axes.Axes, optional
        The axes to plot on. If None, a new figure and axes is created.
    grid_bins : int, optional
        The number of bins to use for the density. Defaults to 200.
    cv_alpha : float, optional
        The alpha value to use for the cross-validation folds. Defaults
        to 0.5.
    cv_line_width : float, optional
        The width of the line to use for the cross-validation folds.
        Defaults to 0.5.
    t_test_alpha : float, optional
        The alpha value to use for the t-test. Defaults to 0.05.
    figsize : tuple, optional
        The size of the figure to create. Defaults to (7, 7). Only used
        if ax is None.
    call_legend : bool, optional
        Whether to call plt.legend() at the end of the function.
        Defaults to True.
    time_series_validation : bool, optional
        Whether the cross-validation is based on a time series. Defaults
        to True.

    Returns
    -------
    matplotlib.axes.Axes
        The axes the plot was drawn on.
    """
    if ax is None:
        _, ax0 = plt.subplots(figsize=figsize)
    else:
        ax0 = ax

    # Convert to pandas Series
    x = to_pd_s(x)
    plot_by = to_pd_s(plot_by)

    cv_label_ = to_pd_s(cv_label)  # Convert cv_label to a pandas Series

    # Plot the density of x by the levels of plot_by
    ax0 = density_by_mpl(x, plot_by, fill_under=False, ax=ax0)

    # Plot the density of x by the levels of plot_by for each cv fold
    ax0 = density_by_mpl(
        x,
        plot_by,
        cv_fold=cv_label_,
        ax=ax0,
        use_labels=False,
        alpha=cv_alpha,
        fill_under=False,
        line_width=cv_line_width,
        time_series_validation=time_series_validation,
    )

    # Add vertical lines at the means and medians of the densities:
    ax0 = _annotate_mean_median(x, plot_by, ax0)

    # Calculate the t-test results:
    _, p, significance_statement = _density_t_test_binary_target(
        x, plot_by, t_test_alpha
    )

    # Add a significance annotation for the t-test results
    ax0 = _significance_annotation(significance_statement, ax0, figsize)

    # Add a dynamic title reflecting the t-test results
    ax0.set_title(
        _get_title(x, plot_by, p, t_test_alpha), fontsize=24 * (figsize[0] / 16)
    )

    # Set the x-axis label
    ax0.set_xlabel(plot_label(x.name if x.name is not None else "Var", False))  # type: ignore

    # show gridlines
    ax0.grid(True)

    if call_legend:
        plt.legend(fontsize=24 * (figsize[0] / 16))

    return ax0


def _significance_annotation(
    significance_statement: str, ax0: plt.Axes, figsize: Tuple[int, int]
) -> plt.Axes:
    ax0.annotate(
        significance_statement,
        xy=(0.5, 0.5),
        xycoords="axes fraction",
        xytext=(0.59, 0.77),
        textcoords="axes fraction",
        ha="left",
        va="center",
        fontsize=24 * (figsize[0] / 16),
        bbox={
            "boxstyle": "round,pad=0.3",
            "edgecolor": "lightgrey",
            "facecolor": "white",
            "alpha": 0.5,
        },
    )
    return ax0


def _get_title(x: pd.Series, plot_by: pd.Series, p: float, t_test_alpha: float) -> str:
    """
    Get the title for the density plot. Helper function for density_plot_mpl.

    Parameters
    ----------
    x : pd.Series
        The variable to plot the density of.
    plot_by : pd.Series
        The variable to group by.
    p : float
        The p-value of the t-test.
    t_test_alpha : float
        The alpha value to use for the t-test.

    Returns
    -------
    str
        The title for the density plot.
    """
    # Add a title reflecting the t-test results
    title = (
        "Kernel Density Plot of "
        f"{plot_label(x.name if x.name is not None else 'Var')}"  # type: ignore
        " by "
        f"{plot_label(plot_by.name if plot_by.name is not None else 'Groupby Var')}."  # type: ignore
    )
    title += "\nDistributions by level are"
    title += " not " if p >= t_test_alpha else " "
    title += f"significantly different at the {1 - t_test_alpha:.0%} level."
    return title


def _density_t_test_binary_target(
    x: Union[pd.Series, pl.Series],
    plot_by: Union[pd.Series, pl.Series],
    alpha: float = 0.05,
):
    """
    Perform a t-test on the density of x by the levels of plot_by. This function
    will be used only when the target variable is binary.

    Parameters
    ----------
    x : Union[pd.Series, pl.Series]
        The variable to plot the density of.
    plot_by : Union[pd.Series, pl.Series]
        The variable to group by. For a binary target, this is the target.
        The plot will generate a density for each level of the target.
    alpha : float, optional
        The alpha value to use for the t-test. Defaults to 0.05.

    Returns
    -------
    t : float
        The t-statistic of the t-test.
    p : float
        The p-value of the t-test.
    significance_statement : str
        A statement indicating whether or not the t-test indicates that the
        distributions are different.
    """
    # Validate inputs
    if not isinstance(x, pd.Series) and not isinstance(x, pl.Series):
        raise ValueError(f"x must be a pandas or polars Series, but got {type(x)}.")
    if not isinstance(plot_by, pd.Series) and not isinstance(plot_by, pl.Series):
        raise ValueError(
            f"plot_by must be a pandas or polars Series, but got {type(plot_by)}."
        )

    # Convert to pandas Series
    x = to_pd_s(x)
    plot_by = to_pd_s(plot_by)

    # Validate that the target is binary
    if get_column_dtype(plot_by) != "binary":
        raise ValueError(
            f"plot_by must be a binary variable, but got {get_column_dtype(plot_by)}."
        )

    # Run a t-test to test whether or not the distributions are the same:
    t, p = ttest_ind(x[plot_by == 0], x[plot_by == 1], equal_var=False)

    # Determine if the distributions are different
    significance_statement = "Results of a Student's t-test:\n=================\n\n"
    if p < alpha:
        significance_statement += "The test indicates that the\ndistributions "
        significance_statement += (
            f"are significantly\ndifferent (p={p:.1e})."
            if p < 1e-3
            else f"are significantly\ndifferent (p={p:.3f})."
        )
    else:
        significance_statement += "The test indicates no significant\ndifference "
        significance_statement += (
            f"between the distributions\n(p={p:.3f}) at the {1-alpha:.0%} level."
        )

    return t, p, significance_statement


def _plot_density_mpl(
    x: pd.Series,
    x_min: Optional[float] = None,
    x_max: Optional[float] = None,
    ax: Optional[Axes] = None,
    label: Union[str, None] = None,
    grid_bins: int = 200,
    line_width: float = 1,
    line_color: Union[str, None] = None,
    line_style: str = "-",
    alpha: float = 1,
    fill_under: bool = True,
    fill_alpha: float = 0.3,
    figsize: Tuple[int, int] = (7, 7),
    time_series_validation: bool = True,
) -> Axes:
    """
    Plot the density of x.

    Parameters
    ----------
    x : pd.Series
        The variable to plot the density of.
    x_min : float, optional
        The minimum value to plot. If None, defaults to the minimum of x. Used
        to extend the curve to the edges of the plot.
    x_max : float, optional
        The maximum value to plot. If None, defaults to the maximum of x. Used
        to extend the curve to the edges of the plot.
    ax : matplotlib.axes.Axes, optional
        The axes to plot on. If None, a new figure and axes is created.
    label : str, optional
        The label for the plot. If None, no label is used.
    grid_bins : int, optional
        The number of bins to use for the density. Defaults to 200.
    line_width : float, optional
        The width of the line to use for the density. Defaults to 1.
    line_style: str, optional
        The style of the line to use for the density. Defaults to '-'. See
        https://matplotlib.org/3.1.0/gallery/lines_bars_and_markers/linestyles.html
        for more information.
    alpha : float, optional
        Global alpha value to use for the plot. Defaults to 1.
    fill_under : bool, optional
        Whether to fill under the density curve. Defaults to True.
    fill_alpha : float, optional
        The alpha value to use for the fill. Defaults to 0.3.
    figsize : tuple, optional
        The size of the figure to create. Defaults to (7, 7). Only used if ax is None.
    time_series_validation : bool, optional
        Whether the cross-validation is based on a time series. Defaults to True.

    Returns
    -------
    matplotlib.axes.Axes
        The axes the plot was drawn on.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    x = x.dropna()

    if x_min is None:
        x_min = x.min()
    if x_max is None:
        x_max = x.max()
    density = gaussian_kde(x)
    x_grid = np.linspace(x_min, x_max, grid_bins)

    if label is None:
        label = x.name if x.name is not None else "Var"  # type: ignore

    if fill_under:
        ax.plot(
            x_grid,
            density(x_grid),
            linewidth=line_width,
            color=line_color,
            ls=line_style,
            alpha=alpha,
        )  # don't label the plot if we're filling under
        ax.fill_between(x_grid, density(x_grid), alpha=fill_alpha, label=label)
    else:
        ax.plot(
            x_grid,
            density(x_grid),
            label=label,
            linewidth=line_width,
            color=line_color,
            ls=line_style,
            alpha=alpha,
        )

    return ax


def density_by_mpl(
    x: pd.Series,
    by: pd.Series,
    cv_fold: Optional[pd.Series] = None,
    x_min: Optional[float] = None,
    x_max: Optional[float] = None,
    ax: Optional[Axes] = None,
    use_labels: bool = True,
    grid_bins: int = 200,
    line_width: float = 1.0,
    alpha: float = 1.0,
    fill_under: bool = True,
    fill_alpha: float = 0.3,
    figsize: Tuple[int, int] = (7, 7),
    time_series_validation: bool = True,
):
    """
    Plot the density of x by the levels of by, using matplotlib,
    all on the same axss.

    Parameters
    ----------
    x : pd.Series
        The variable to plot the density of.
    by : pd.Series
        The variable to group by.
    cv_fold : pd.Series, optional
        The cross-validation fold to group by. If None, no grouping is done.
        Defaults to None.
    x_min : float, optional
        The minimum value to plot. If None, defaults to the minimum of x. Used
        to extend the curve to the edges of the plot.
    x_max : float, optional
        The maximum value to plot. If None, defaults to the maximum of x. Used
        to extend the curve to the edges of the plot.
    ax : matplotlib.axes.Axes, optional
        The axes to plot on. If None, a new figure and axes is created.
    use_labels : bool, optional
        Whether to use the labels of by. Defaults to True.
    grid_bins : int, optional
        The number of bins to use for the density. Defaults to 200. Controls
        how smooth the density plot is.
    line_width : float, optional
        The width of the line to use for the density. Defaults to 1.
    line_color : str, optional
        The color of the line to use for the density. Defaults to None.
    alpha : float, optional
        Global alpha value to use for the plot. Defaults to 1.
    fill_under : bool, optional
        Whether to fill under the density curve. Defaults to True.
    fill_alpha : float, optional
        The alpha value to use for the fill. Defaults to 0.3. Does nothing if
        fill_under is False.
    figsize : tuple, optional
        The size of the figure to create. Defaults to (7, 7). Only used if ax is None.
    time_series_validation : bool, optional
        Whether the cross-validation is based on a time series. Defaults to True.

    Returns
    -------
    matplotlib.axes.Axes
        The axes the plot was drawn on.

    Not Yet Implemented
    -------------------
    1. I want to have the CV curves calculate a standard deviation, and provide a
       +/- 1 SD band.
    """
    # If no axis passed, create one
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    # Set min and max if not set, and build params dict
    x_min, x_max = graph_min_max(x, x_min, x_max)
    params = {
        "x_min": x_min,
        "x_max": x_max,
        "ax": ax,
        "grid_bins": grid_bins,
        "line_width": line_width,
        "alpha": alpha,
        "fill_under": fill_under,
        "fill_alpha": fill_alpha,
        "figsize": figsize,
    }

    # If not grouping by CV fold, just plot the density
    if cv_fold is None:
        for level, group in x.groupby(by):
            byname = by.name if by.name is not None else "Groupby Var"
            color_ = binary_color(level) if get_column_dtype(by) == "binary" else None
            color__ = color_ if color_ is not None else None
            label = f"{plot_label(byname)} = {level}"  # type: ignore
            _plot_density_mpl(
                group,
                label=label,
                line_color=color__,
                **params,  # type: ignore
            )
    # Otherwise, plot the density by CV fold

    else:
        for f in cv_fold.drop_duplicates().sort_values():  # loop over CV fold
            x_ = filter_by_cv_fold(x, f, cv_fold, time_series_validation, "test")
            by_ = filter_by_cv_fold(by, f, cv_fold, time_series_validation, "test")
            for level, group in x_.groupby(by_):  # type: ignore
                color_ = (
                    binary_color(level) if get_column_dtype(by) == "binary" else None
                )
                color__ = color_ if color_ is not None else None
                _plot_density_mpl(
                    group,
                    label="_nolegend_",
                    line_color=color__,
                    **params,  # type: ignore
                )

    return ax


def calculate_density_sd(
    x: Union[pd.Series, pl.Series, np.ndarray],
    by: Union[pd.Series, pl.Series, np.ndarray],
    cv_fold: Union[pd.Series, pl.Series, np.ndarray],
    x_min: Optional[float] = None,
    x_max: Optional[float] = None,
    grid_bins: int = 200,
    time_series_validation: bool = True,
):
    """
    Calculates the standard deviation of the kernel density estimates (KDE) of `x`
    grouped by the levels of `by` across different cross-validation folds specified
    by `cv_fold`.

    This function computes the KDE of `x` for each group defined by unique values
    in `by` for each cross-validation fold. It then calculates the standard deviation
    of these density estimates at a fixed set of points defined by `grid_bins`. The
    result is a smoothed series of standard deviations representing the variability
    of the density estimate across folds and groups.

    Parameters
    ----------
    x : Union[pd.Series, pl.Series, np.ndarray]
        The data series for which the density's standard deviation is computed.
    by : Union[pd.Series, pl.Series, np.ndarray]
        A series indicating the group of each observation in `x`.
    cv_fold : Union[pd.Series, pl.Series, np.ndarray]
        A series indicating the cross-validation fold of each observation in `x`.
        Unique values in `cv_fold` are treated as separate folds.
    x_min : float, optional
        The minimum value to plot. If None, defaults to the minimum of x before
        grouping by any variables. Used to extend the curve to the edges of the plot.
    x_max : float, optional
        The maximum value to plot. If None, defaults to the maximum of x before
        grouping by any variables. Used to extend the curve to the edges of the plot.
    grid_bins : int, optional
        The number of points at which to evaluate the density and its standard
        deviation. Defaults to 200.
    time_series_validation : bool, optional
        Indicates whether the cross-validation is based on a time series split.
        This parameter influences how the data is split into training and testing
        sets. Defaults to True.

    Returns
    -------
    tuple of pd.Series
        A tuple containing two pandas Series:
        - The first Series contains the smoothed standard deviations of the density
          estimates across the cross-validation folds and groups.
        - The second Series contains the unsmoothed standard deviations.

    Notes
    -----
    Smoothing is applied by averaging the standard deviations within a sliding
    window of 5 points, with special handling for the first two and last two points
    to avoid boundary effects.
    """
    # Convert to pandas Series
    x = to_pd_s(x)
    by = to_pd_s(by)
    cv_fold = to_pd_s(cv_fold)

    # Raise an error if either `x`, `by`, or `cv_fold` is empty
    if x.shape[0] == 0:
        raise ValueError("The input series `x` cannot be empty.")
    if by.shape[0] == 0:
        raise ValueError("The input series `by` cannot be empty.")
    if cv_fold.shape[0] == 0:
        raise ValueError("The input series `cv_fold` cannot be empty.")

    xmin, xmax = graph_min_max(x, x_min, x_max)

    sd = pd.DataFrame({"x": np.linspace(xmin, xmax, grid_bins)}, index=range(grid_bins))
    for f in cv_fold.drop_duplicates().sort_values():
        x_ = filter_by_cv_fold(x, f, cv_fold, time_series_validation, "test")
        by_ = filter_by_cv_fold(by, f, cv_fold, time_series_validation, "test")
        for level, group in x_.groupby(by_):  # type: ignore
            density = gaussian_kde(group)
            sd[f"{f}_{level}"] = density(sd["x"])

    sd = sd.drop(columns=["x"])
    sd = sd.std(axis=1).iloc[:, 0] if len(sd.columns) == 1 else sd.std(axis=1)  # type: ignore

    # smooth the standard deviation (should not deviate much from one
    # x value to the next)
    sd_smooth = sd.rolling(window=5, center=True).mean()
    sd_smooth[0] = np.mean(sd[:2])
    sd_smooth[1] = np.mean(sd[:3])

    sd_smooth[len(sd_smooth) - 1] = np.mean(sd[-2:])
    sd_smooth[len(sd_smooth) - 2] = np.mean(sd[-3:])

    return sd_smooth, sd


def _calculate_single_density_sd(
    x: Union[pd.Series, pl.Series, np.ndarray],
    cv_fold: Union[pd.Series, pl.Series, np.ndarray],
    grid_bins: int = 200,
    time_series_validation: bool = True,
) -> pd.Series:
    """
    Calculates the standard deviation of the density estimates of `x` across
    different cross-validation folds.

    This function computes the kernel density estimate (KDE) of `x` for each unique
    value in `cv_fold`. It then calculates the standard deviation of these density
    estimates at a fixed set of points defined by `grid_bins`. The result is a
    smoothed series of standard deviations representing the variability of the
    density estimate across folds.

    Parameters
    ----------
    x : Union[pd.Series, pl.Series, np.ndarray]
        The data series for which the density's standard deviation is to be computed.
    cv_fold : Union[pd.Series, pl.Series, np.ndarray]
        A series indicating the cross-validation fold of each observation in `x`.
        Unique values in `cv_fold` are treated as separate folds.
    grid_bins : int, optional
        The number of points at which to evaluate the density and its standard
        deviation. Defaults to 200.
    time_series_validation : bool, optional
        Indicates whether the cross-validation is based on a time series split.
        This parameter is passed to `filter_by_cv_fold` to determine how the data
        is split into training and testing sets. Defaults to True.

    Returns
    -------
    pd.Series
        A pandas Series containing the smoothed standard deviations of the density
        estimates across the cross-validation folds.

    Notes
    -----
    The smoothing is applied by averaging the standard deviations within a sliding
    window of 5 points, with special handling for the first two and last two points
    to avoid boundary effects.
    """
    # Convert to pandas Series
    x = to_pd_s(x)
    cv_fold = to_pd_s(cv_fold)

    # Raise an error if either `x` or `cv_fold` is empty
    if x.shape[0] == 0:
        raise ValueError("The input series `x` cannot be empty.")
    if cv_fold.shape[0] == 0:
        raise ValueError("The input series `cv_fold` cannot be empty.")

    sd = pd.DataFrame(
        {"x": np.linspace(x.min(), x.max(), grid_bins)}, index=range(grid_bins)
    )
    for f in cv_fold.drop_duplicates().sort_values():
        x_ = filter_by_cv_fold(x, f, cv_fold, time_series_validation, "test")
        density = gaussian_kde(x_)
        sd[f"{f}"] = density(sd["x"])

    sd = sd.drop(columns=["x"])
    sd = sd.std(axis=1).iloc[:, 0] if len(sd.columns) == 1 else sd.std(axis=1)  # type: ignore

    # smooth the standard deviation (should not deviate much from one
    # x value to the next)
    sd_smooth = sd.rolling(window=5, center=True).mean()
    sd_smooth[0] = np.mean(sd[:2])
    sd_smooth[1] = np.mean(sd[:3])

    sd_smooth[len(sd_smooth) - 1] = np.mean(sd[-2:])
    sd_smooth[len(sd_smooth) - 2] = np.mean(sd[-3:])

    return sd_smooth, sd  # type: ignore


def _plot_single_density_pm_standard_deviation(
    x: pd.Series,
    cv_fold: Union[pd.Series, None] = None,
    x_min: Optional[float] = None,
    x_max: Optional[float] = None,
    ax: Optional[Axes] = None,
    label: Union[str, None] = None,
    grid_bins: int = 200,
    line_width: float = 0.5,
    line_style: str = "-",
    alpha: float = 0.5,
    fill_alpha: float = 0.3,
    figsize: Tuple[int, int] = (7, 7),
    time_series_validation: bool = True,
) -> Axes:
    """
    Plot the density of x.

    Parameters
    ----------
    x : pd.Series
        The variable to plot the density of.
    sd: pd.Series
        The standard deviation of the density of x. Has the same shape as x.
    x_min : float, optional
        The minimum value to plot. If None, defaults to the minimum of x. Used
        to extend the curve to the edges of the plot.
    x_max : float, optional
        The maximum value to plot. If None, defaults to the maximum of x. Used
        to extend the curve to the edges of the plot.
    ax : matplotlib.axes.Axes, optional
        The axes to plot on. If None, a new figure and axes is created.
    label : str, optional
        The label for the plot. If None, no label is used.
    grid_bins : int, optional
        The number of bins to use for the density. Defaults to 200.
    line_width : float, optional
        The width of the line to use for the density. Defaults to 1.
    line_style: str, optional
        The style of the line to use for the density. Defaults to '-'. See
        https://matplotlib.org/3.1.0/gallery/lines_bars_and_markers/linestyles.html
        for more information.
    alpha : float, optional
        Global alpha value to use for the plot. Defaults to 1.
    fill_alpha : float, optional
        The alpha value to use for the fill. Defaults to 0.3.
    figsize : tuple, optional
        The size of the figure to create. Defaults to (7, 7). Only used if ax is None.
    time_series_validation : bool, optional
        Whether the cross-validation is based on a time series. Defaults to True.

    Returns
    -------
    matplotlib.axes.Axes
        The axes the plot was drawn on.
    """
    if ax is None:
        _, ax_ = plt.subplots(figsize=figsize)
    else:
        ax_ = ax

    x = x.dropna()

    # Set min and max if not set
    x_min = x.min() if x_min is None else x_min
    x_max = x.max() if x_max is None else x_max

    # Calculate density
    density = gaussian_kde(x)

    # Create grid for a smooth plot and calculate density
    x_grid = np.linspace(x_min, x_max, grid_bins)

    df = pd.DataFrame({"x": x_grid, "density": density(x_grid)})

    df1 = pd.DataFrame({"x": np.linspace(df.x.min(), df.x.max(), grid_bins)})
    df1["density"] = density(np.linspace(df.x.min(), df.x.max(), grid_bins))

    return ax_


def _annotate_mean_median(
    feature: Union[pd.Series, pl.Series, np.ndarray],
    target: Union[pd.Series, pl.Series, np.ndarray],
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Annotates the mean and median of the feature variable for each target class.

    Puts a box around the annotations with a blue/orange background and black
    border. The boxes are offset to either the left or right of the mean/median
    depending on whether mean0 < mean1 or mean0 > mean1.

    Extending from the boxes are arrows pointing at the mean and median lines. The
    arrows are on lines that are curving in opposite directions, the arrow pointing
    to curving up for the mean and curving down for the median. The arrows are
    annotated with the Mean/Median ratios for each target class.

    Parameters
    ----------
    feature : Union[pd.Series, pl.Series, np.ndarray]
        The feature variable.
    target : Union[pd.Series, pl.Series, np.ndarray]
        The target variable.
    ax : plt.Axes, optional
        The axes to annotate. If None, a new figure and axes is created.

    Returns
    -------
    plt.Axes
        The axes with the annotations added.
    """
    # Raise an error if the feature and target are not the same length
    if len(feature) != len(target):
        raise ValueError(
            "The feature and target variables must be the same length, "
            f"but got {len(feature)} and {len(target)}."
        )
    elif (len(feature) == 0) | (len(target) == 0):
        raise ValueError(
            "The feature and target series should not contain NaN or missing values"
        )

    # Raise an error if either the feature or target contain NaNs
    if (
        isinstance(feature, pd.Series)
        and feature.isnull().any()
        or isinstance(feature, pl.Series)
        and feature.is_null().any()
        or isinstance(feature, np.ndarray)
        and np.isnan(feature).any()
    ):
        raise ValueError(
            "The feature and target series should not contain NaN or missing values"
        )

    if (
        isinstance(target, pd.Series)
        and target.isnull().any()
        or isinstance(target, pl.Series)
        and target.is_null().any()
        or isinstance(target, np.ndarray)
        and np.isnan(target).any()
    ):
        raise ValueError(
            "The feature and target series should not contain NaN or missing values"
        )

    # Convert to pandas Series
    feature_ = to_pd_s(feature)
    target_ = to_pd_s(target)

    # Calculate means and medians
    mean0, mean1 = feature_[target_ == 0].mean(), feature_[target_ == 1].mean()
    median0, median1 = (
        feature_[target_ == 0].median(),
        feature_[target_ == 1].median(),
    )

    # If no axis passed, create one
    ax_ = plt.subplots()[1] if ax is None else ax

    # Add vertical lines
    ax_.axvline(mean0, color="blue", linestyle="--", linewidth=1)
    ax_.axvline(mean1, color="orange", linestyle="--", linewidth=1)
    ax_.axvline(median0, color="blue", linestyle="dotted", linewidth=1)
    ax_.axvline(median1, color="orange", linestyle="dotted", linewidth=1)

    # Define annotation position and arrow properties based on mean0 and mean1
    pos0, pos1 = ("right", "left") if mean0 < mean1 else ("left", "right")
    arrowprops0 = {"arrowstyle": "->", "lw": 1}
    arrowprops1 = {"arrowstyle": "->", "lw": 1}

    # Extract the figure size
    figsize = ax_.get_figure().get_size_inches()  # type: ignore

    # Annotate for target=0
    ax_.annotate(
        f"{target_.name}=0\n===========\nMean / Median =\n{mean0 / (median0 if median0 != 0 else 1):.2f}",
        xy=(mean0, 0.2),
        xycoords="data",
        xytext=(-20 if pos0 == "right" else 20, -20),
        textcoords="offset points",
        ha=pos0,
        va="bottom",
        fontsize=24 * (figsize[0] / 16),
        bbox={
            "boxstyle": "round,pad=0.3",
            "edgecolor": "black",
            "facecolor": "blue",
            "alpha": 0.2,
        },
        arrowprops=arrowprops0,
    )

    # Annotate for target=1
    ax_.annotate(
        f"{target_.name}=1\n===========\nMean / Median =\n{mean1 / (median1 if median1 != 0 else 1):.2f}",
        xy=(mean1, 0.2),
        xycoords="data",
        xytext=(-20 if pos1 == "right" else 20, -20),
        textcoords="offset points",
        ha=pos1,
        va="bottom",
        fontsize=24 * (figsize[0] / 16),
        bbox={
            "boxstyle": "round,pad=0.3",
            "edgecolor": "black",
            "facecolor": "orange",
            "alpha": 0.2,
        },
        arrowprops=arrowprops1,
    )

    return ax_
