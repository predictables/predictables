from typing import Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
from matplotlib.axes import Axes
from scipy.stats import gaussian_kde, ttest_ind  # type: ignore

from predictables.univariate.src.plots.util import binary_color, plot_label
from predictables.util import get_column_dtype, to_pd_s


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
) -> Axes:
    """
    Plot density function as well as cross-validation densities using the specified backend.

    Parameters
    ----------
    x : Union[pd.Series, pl.Series]
        The variable to plot the density of.
    plot_by : Union[pd.Series, pl.Series]
        The variable to group by. For a binary target, this is the target. The plot will
        generate a density for each level of the target.
    cv_label : Union[pd.Series, pl.Series]
        The cross-validation fold to group by. If None, no grouping is done. Defaults to None.
    x_min : float, optional
        The minimum value to plot. If None, defaults to the minimum of x before grouping by
        any variables. Used to extend the curve to the edges of the plot.
    x_max : float, optional
        The maximum value to plot. If None, defaults to the maximum of x before grouping by
        any variables. Used to extend the curve to the edges of the plot.
    ax : matplotlib.axes.Axes, optional
        The axes to plot on. If None, a new figure and axes is created.
    grid_bins : int, optional
        The number of bins to use for the density. Defaults to 200.
    cv_alpha : float, optional
        The alpha value to use for the cross-validation folds. Defaults to 0.5.
    cv_line_width : float, optional
        The width of the line to use for the cross-validation folds. Defaults to 0.5.
    t_test_alpha : float, optional
        The alpha value to use for the t-test. Defaults to 0.05.
    figsize : tuple, optional
        The size of the figure to create. Defaults to (7, 7). Only used if ax is None.
    call_legend : bool, optional
        Whether to call plt.legend() at the end of the function. Defaults to True.
    backend : str, optional
        The backend to use for plotting. Defaults to "matplotlib".

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
        )
    elif backend == "plotly":
        raise NotImplementedError(
            "Plotly backend not yet implemented. Use 'matplotlib' for now."
        )
    else:
        raise ValueError(f"Invalid backend {backend}.")


# trunk-ignore(sourcery/low-code-quality)
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

    Returns
    -------
    matplotlib.axes.Axes
        The axes the plot was drawn on.

    Raises
    ------
    ValueError
        If plot_by is None.
    """
    # Validate inputs
    if not isinstance(x, pd.Series) and not isinstance(x, pl.Series):
        raise ValueError(
            f"x must be a pandas or polars Series, but got {type(x)}."
        )

    if not isinstance(plot_by, pd.Series) and not isinstance(
        plot_by, pl.Series
    ):
        raise ValueError(
            f"plot_by must be a pandas or polars Series, but got {type(plot_by)}."
        )

    if not isinstance(cv_label, pd.Series) and not isinstance(
        cv_label, pl.Series
    ):
        raise ValueError(
            f"cv_label must be a pandas or polars Series, but got {type(cv_label)}."
        )

    if (
        x_min is not None
        and not isinstance(x_min, float)
        and not isinstance(x_min, int)
    ):
        raise ValueError(
            f"x_min must be a float or int, but got {type(x_min)}."
        )

    if (
        x_max is not None
        and not isinstance(x_max, float)
        and not isinstance(x_max, int)
    ):
        raise ValueError(
            f"x_max must be a float or int, but got {type(x_max)}."
        )

    if not isinstance(grid_bins, int):
        raise ValueError(
            f"grid_bins must be an int, but got {type(grid_bins)}."
        )

    if not isinstance(figsize, tuple):
        raise ValueError(f"figsize must be a tuple, but got {type(figsize)}.")

    if ax is None:
        _, ax0 = plt.subplots(figsize=figsize)
    else:
        ax0 = ax

    # Convert to pandas Series
    x = to_pd_s(x)
    plot_by = to_pd_s(plot_by)

    ax0 = density_by_mpl(x, plot_by, fill_under=False, ax=ax0)
    ax0 = density_by_mpl(
        x,
        plot_by,
        cv_fold=cv_label,
        ax=ax0,
        use_labels=False,
        alpha=cv_alpha,
        fill_under=False,
        line_width=cv_line_width,
    )

    # Add vertical lines at the means and medians of the densities:
    ax0 = _annotate_mean_median(ax0, x, plot_by)

    # Annotate the t-test results:
    _, p, significance_statement = _density_t_test_binary_target(
        x, plot_by, t_test_alpha
    )
    ax0.annotate(
        significance_statement,
        xy=(0.5, 0.5),
        xycoords="axes fraction",
        xytext=(0.59, 0.77),
        textcoords="axes fraction",
        ha="left",
        va="center",
        fontsize=24 * (figsize[0] / 16),
        bbox=dict(
            boxstyle="round,pad=0.3",
            edgecolor="lightgrey",
            facecolor="white",
            alpha=0.5,
        ),
    )

    # Add a title reflecting the t-test results
    title = f"Kernel Density Plot of {plot_label(x.name)} by {plot_label(plot_by.name)}\nDistributions by level are"
    title += " not " if p >= t_test_alpha else " "
    title += f"significantly different at the {1 - t_test_alpha:.0%} level."

    ax0.set_title(title)

    # Set the x-axis label
    ax0.set_xlabel(plot_label(x.name, False))

    if call_legend:
        plt.legend(fontsize=24 * (figsize[0] / 16))

    return ax0


def _density_t_test_binary_target(
    x: Union[pd.Series, pl.Series],
    plot_by: Union[pd.Series, pl.Series],
    alpha: float = 0.05,
):
    """
    Perform a t-test on the density of x by the levels of plot_by. This function will be used only when the target variable is binary.

    Parameters
    ----------
    x : Union[pd.Series, pl.Series]
        The variable to plot the density of.
    plot_by : Union[pd.Series, pl.Series]
        The variable to group by. For a binary target, this is the target. The plot will generate a density for each level of the target.
    alpha : float, optional
        The alpha value to use for the t-test. Defaults to 0.05.

    Returns
    -------
    t : float
        The t-statistic of the t-test.
    p : float
        The p-value of the t-test.
    significance_statement : str
        A statement indicating whether or not the t-test indicates that the distributions are different.
    """
    # Validate inputs
    if not isinstance(x, pd.Series) and not isinstance(x, pl.Series):
        raise ValueError(
            f"x must be a pandas or polars Series, but got {type(x)}."
        )
    if not isinstance(plot_by, pd.Series) and not isinstance(
        plot_by, pl.Series
    ):
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
    significance_statement = (
        "Results of a Student's t-test:\n=================\n\n"
    )
    if p < alpha:
        if p < 1e-3:
            significance_statement += f"The test indicates that the\ndistributions are significantly\ndifferent (p={p:.1e})."
        else:
            significance_statement += f"The test indicates that the\ndistributions are significantly\ndifferent (p={p:.3f})."
    else:
        significance_statement += f"The test indicates no significant\ndifference between the distributions\n(p={p:.3f}) at the {1-alpha:.0%} level."

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
        label = x.name

    if fill_under:
        ax.plot(
            x_grid,
            density(x_grid),
            linewidth=line_width,
            color=line_color,
            ls=line_style,
            alpha=alpha,
        )  # don't label the plot if we're filling under
        ax.fill_between(
            x_grid,
            density(x_grid),
            alpha=fill_alpha,
            label=label,
        )
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
    cv_fold: Optional[Union[pd.Series, None]] = None,
    x_min: Optional[Optional[float]] = None,
    x_max: Optional[Optional[float]] = None,
    ax: Optional[Optional[Axes]] = None,
    use_labels: Optional[bool] = True,
    grid_bins: Optional[int] = 200,
    line_width: Optional[float] = 1.0,
    alpha: Optional[float] = 1.0,
    fill_under: Optional[bool] = True,
    fill_alpha: Optional[float] = 0.3,
    figsize: Optional[Tuple[int, int]] = (7, 7),
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
        The cross-validation fold to group by. If None, no grouping is done. Defaults to None.
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
        The number of bins to use for the density. Defaults to 200. Controls how smooth the density plot is.
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

    Returns
    -------
    matplotlib.axes.Axes
        The axes the plot was drawn on.

    Not Yet Implemented
    -------------------
    1. I want to have the CV curves calculate a standard deviation, and provide a +/- 1 SD
       band.
    """
    # If no axis passed, create one
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    # Set min and max if not set, and build params dict
    x_min = x_min if x_min is not None else x.dropna().min()
    x_max = x_max if x_max is not None else x.dropna().max()
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
            color = (
                binary_color(level)
                if get_column_dtype(by) == "binary"
                else None
            )
            label = f"{plot_label(by.name)} = {level}"
            _plot_density_mpl(
                group,
                label=label,
                line_color=color,
                **params,
            )
    # Otherwise, plot the density by CV fold
    else:
        for f in cv_fold.drop_duplicates().sort_values():  # loop over CV fold
            for level, group in x[cv_fold == f].groupby(by[cv_fold == f]):
                color = (
                    binary_color(level)
                    if get_column_dtype(by) == "binary"
                    else None
                )
                _plot_density_mpl(
                    group,
                    label="_nolegend_",  # don't label the plot if we're filling under
                    line_color=color,
                    **params,
                )

    return ax


def calculate_density_sd(
    x: pd.Series,
    by: pd.Series,
    cv_fold: Union[pd.Series, None] = None,
    grid_bins: int = 200,
):
    """
    Using the cross-validation folds, calculate the standard deviation of the
    density of x by the levels of by.
    """
    if cv_fold is None:
        raise ValueError("cv_fold cannot be None.")

    sd = pd.DataFrame(
        {"x": np.linspace(x.min(), x.max(), grid_bins)}, index=range(grid_bins)
    )
    for f in cv_fold.drop_duplicates().sort_values():
        for level, group in x[cv_fold == f].groupby(by[cv_fold == f]):
            density = gaussian_kde(group)
            sd[f"{f}_{level}"] = density(sd["x"])

    sd = sd.drop(columns=["x"])
    sd = sd.std(axis=1)

    # smooth the standard deviation (should not deviate much from one
    # x value to the next)
    sd_smooth = sd.rolling(window=5, center=True).mean()
    sd_smooth[0] = np.mean(sd[:2])
    sd_smooth[1] = np.mean(sd[:3])

    sd_smooth[len(sd_smooth) - 1] = np.mean(sd[-2:])
    sd_smooth[len(sd_smooth) - 2] = np.mean(sd[-3:])

    return sd_smooth, sd


def _calculate_single_density_sd(
    x: pd.Series,
    cv_fold: pd.Series,
    grid_bins: int = 200,
) -> pd.Series:
    """
    Using the cross-validation folds, calculate the standard deviation of the
    density of x.
    """
    sd = pd.DataFrame(
        {"x": np.linspace(x.min(), x.max(), grid_bins)}, index=range(grid_bins)
    )
    for f in cv_fold.drop_duplicates().sort_values():
        density = gaussian_kde(x[cv_fold == f])
        sd[f"{f}"] = density(sd["x"])

    sd = sd.drop(columns=["x"])
    sd = sd.std(axis=1)

    # smooth the standard deviation (should not deviate much from one
    # x value to the next)
    sd_smooth = sd.rolling(window=5, center=True).mean()
    sd_smooth[0] = np.mean(sd[:2])
    sd_smooth[1] = np.mean(sd[:3])

    sd_smooth[len(sd_smooth) - 1] = np.mean(sd[-2:])
    sd_smooth[len(sd_smooth) - 2] = np.mean(sd[-3:])

    return sd_smooth, sd


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

    Returns
    -------
    matplotlib.axes.Axes
        The axes the plot was drawn on.
    """

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    x = x.dropna()

    # Set min and max if not set
    if x_min is None:
        x_min = x.min()
    if x_max is None:
        x_max = x.max()

    # Calculate density
    density = gaussian_kde(x)

    # Create grid for a smooth plot and calculate density
    x_grid = np.linspace(x_min, x_max, grid_bins)

    df = pd.DataFrame({"x": x_grid, "density": density(x_grid)})

    df1 = pd.DataFrame({"x": np.linspace(df.x.min(), df.x.max(), grid_bins)})
    df1["density"] = density(np.linspace(df.x.min(), df.x.max(), grid_bins))

    return ax


def _annotate_mean_median(
    ax: plt.Axes, feature: pd.Series, target: pd.Series
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

    Parameters:
    -----------
    ax (matplotlib.axes.Axes): The axis to add the annotations to.
    feature (pandas.Series): The feature variable data.
    target (pandas.Series): The target variable data.

    Returns:
    --------
    ax (matplotlib.axes.Axes): The axis with the annotations added.
    """
    # Calculate means and medians
    mean0, mean1 = feature[target == 0].mean(), feature[target == 1].mean()
    median0, median1 = (
        feature[target == 0].median(),
        feature[target == 1].median(),
    )

    # Add vertical lines
    ax.axvline(mean0, color="blue", linestyle="--", linewidth=1)
    ax.axvline(mean1, color="orange", linestyle="--", linewidth=1)
    ax.axvline(median0, color="blue", linestyle="dotted", linewidth=1)
    ax.axvline(median1, color="orange", linestyle="dotted", linewidth=1)

    # Define annotation position and arrow properties based on mean0 and mean1
    pos0, pos1 = ("right", "left") if mean0 < mean1 else ("left", "right")
    arrowprops0 = dict(arrowstyle="->", lw=1)
    arrowprops1 = dict(arrowstyle="->", lw=1)

    # Extract the figure size
    figsize = ax.get_figure().get_size_inches()

    # Annotate for target=0
    ax.annotate(
        f"{target.name}=0\n===========\nMean / Median =\n{mean0 / median0:.2f}",
        xy=(mean0, 0.2),
        xycoords="data",
        xytext=(-20 if pos0 == "right" else 20, -20),
        textcoords="offset points",
        ha=pos0,
        va="bottom",
        fontsize=24 * (figsize[0] / 16),
        bbox=dict(
            boxstyle="round,pad=0.3",
            edgecolor="black",
            facecolor="blue",
            alpha=0.2,
        ),
        arrowprops=arrowprops0,
    )

    # Annotate for target=1
    ax.annotate(
        f"{target.name}=1\n===========\nMean / Median =\n{mean1 / median1:.2f}",
        xy=(mean1, 0.2),
        xycoords="data",
        xytext=(-20 if pos1 == "right" else 20, -20),
        textcoords="offset points",
        ha=pos1,
        va="bottom",
        fontsize=24 * (figsize[0] / 16),
        bbox=dict(
            boxstyle="round,pad=0.3",
            edgecolor="black",
            facecolor="orange",
            alpha=0.2,
        ),
        arrowprops=arrowprops1,
    )

    return ax
