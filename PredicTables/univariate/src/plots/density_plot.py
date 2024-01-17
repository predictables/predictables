from typing import Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from scipy.stats import gaussian_kde

from PredicTables.univariate.src.plots.util import plot_label
from PredicTables.util import get_column_dtype


def binary_color(x: int) -> pd.Series:
    """
    Return a color for each value in x, based on whether it is 0 or 1.

    Parameters
    ----------
    x : int
        The value to get the color for.

    Returns
    -------
    int
        The color for x.
    """
    if x == 0:
        return "blue"
    elif x == 1:
        return "orange"
    else:
        raise ValueError(f"Invalid value {x} for binary variable.")


def _plot_density_mpl(
    x: pd.Series,
    x_min: Union[float, None] = None,
    x_max: Union[float, None] = None,
    ax: Union[Axes, None] = None,
    label: Union[str, None] = None,
    grid_bins: int = 200,
    line_width: float = 1,
    line_style: str = "-",
    alpha: float = 1,
    fill_under: bool = True,
    fill_alpha: float = 0.3,
    figsize: Tuple[int, int] = (8, 8),
    **kwargs,
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
        The size of the figure to create. Defaults to (8, 8). Only used if ax is None.
    **kwargs
        Additional keyword arguments passed to ax.plot.

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

    if fill_under:
        ax.plot(
            x_grid,
            density(x_grid),
            linewidth=line_width,
            ls=line_style,
            alpha=alpha,
            **kwargs,
        )  # don't label the plot if we're filling under
        ax.fill_between(
            x_grid, density(x_grid), alpha=fill_alpha, label=label, **kwargs
        )
    else:
        ax.plot(
            x_grid,
            density(x_grid),
            label=label,
            linewidth=line_width,
            ls=line_style,
            alpha=alpha,
            **kwargs,
        )

    return ax


def density_by_mpl(
    x: pd.Series,
    by: pd.Series,
    cv_fold: Union[pd.Series, None] = None,
    x_min: Union[float, None] = None,
    x_max: Union[float, None] = None,
    ax: Union[Axes, None] = None,
    use_labels: bool = True,
    grid_bins: int = 200,
    line_width: float = 1,
    alpha: float = 1,
    fill_under: bool = True,
    fill_alpha: float = 0.3,
    figsize: Tuple[int, int] = (8, 8),
    **kwargs,
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
        The number of bins to use for the density. Defaults to 200. Controls
        how smooth the density plot is.
    line_width : float, optional
        The width of the line to use for the density. Defaults to 1.
    alpha : float, optional
        Global alpha value to use for the plot. Defaults to 1.
    fill_under : bool, optional
        Whether to fill under the density curve. Defaults to True.
    fill_alpha : float, optional
        The alpha value to use for the fill. Defaults to 0.3. Does nothing if
        fill_under is False.
    figsize : tuple, optional
        The size of the figure to create. Defaults to (8, 8). Only used if ax is None.
    **kwargs
        Additional keyword arguments passed to ax.plot.

    Returns
    -------
    matplotlib.axes.Axes
        The axes the plot was drawn on.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    if x_min is None:
        x_min = x.dropna().min()
    if x_max is None:
        x_max = x.dropna().max()

    if cv_fold is None:
        for level, group in x.groupby(by):
            if get_column_dtype(by) == "binary":
                color = binary_color(level)
            else:
                color = None
            label = f"{plot_label(by.name)} = {level}"
            _plot_density_mpl(
                group,
                x_min=x_min,
                x_max=x_max,
                ax=ax,
                label=(label if use_labels else None),
                grid_bins=grid_bins,
                line_width=line_width,
                alpha=alpha,
                fill_under=fill_under,
                fill_alpha=fill_alpha,
                figsize=figsize,
                color=color,
                **kwargs,
            )
    else:
        for f in cv_fold.drop_duplicates().sort_values():
            for level, group in x[cv_fold == f].groupby(by[cv_fold == f]):
                if get_column_dtype(by) == "binary":
                    color = binary_color(level)
                else:
                    color = None
                label = f"{plot_label(by.name)}(Fold {f}) = {level}"
                _plot_density_mpl(
                    group,
                    x_min=x_min,
                    x_max=x_max,
                    ax=ax,
                    label=(label if use_labels else None),
                    grid_bins=grid_bins,
                    line_width=line_width,
                    alpha=alpha,
                    fill_under=fill_under,
                    fill_alpha=fill_alpha,
                    figsize=figsize,
                    color=color,
                    **kwargs,
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
    else:
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
        # return pd.DataFrame(
        #     {
        #         "x": np.linspace(x.min(), x.max(), grid_bins),
        #         "sd": sd_smooth,
        #         "raw_sd": sd,
        #     }
        # )
        return sd_smooth


def _plot_single_density_pm_standard_deviation(
    x: pd.Series,
    sd: pd.Series,
    x_min: Union[float, None] = None,
    x_max: Union[float, None] = None,
    ax: Union[Axes, None] = None,
    label: Union[str, None] = None,
    grid_bins: int = 200,
    line_width: float = 0.5,
    line_style: str = "-",
    alpha: float = 0.5,
    fill_alpha: float = 0.3,
    figsize: Tuple[int, int] = (8, 8),
    **kwargs,
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
        The size of the figure to create. Defaults to (8, 8). Only used if ax is None.
    **kwargs
        Additional keyword arguments passed to ax.plot.

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

    df = pd.DataFrame({"x": x_grid, "density": density(x_grid)})
    df["sd"] = sd
    df["lower"] = df["density"] - df["sd"]
    df["upper"] = df["density"] + df["sd"]

    # plot the outlines of the density +/- the standard deviation
    ax.plot(
        df.x,
        df.lower,
        linewidth=line_width,
        ls=line_style,
        alpha=alpha,
        label=None,
        **kwargs,
    )  # don't label the plot if we're filling under
    ax.plot(
        df.x,
        df.upper,
        linewidth=line_width,
        ls=line_style,
        alpha=alpha,
        label=None,
        **kwargs,
    )  # don't label the plot if we're filling under
    ax.fill_between(df.x, df.lower, df.upper, alpha=fill_alpha, label=label, **kwargs)

    return ax


def plot_density_pm_standard_deviation(
    x: pd.Series,
    x_min: Union[float, None] = None,
    x_max: Union[float, None] = None,
    by: Union[pd.Series, None] = None,
    cv_fold: Union[pd.Series, None] = None,
    grid_bins: int = 200,
    ax: Union[Axes, None] = None,
    use_labels: bool = True,
    line_width: float = 0.5,
    alpha: float = 0.5,
):
    """
    Plot a shaded region around the density of x by the levels of by, using
    matplotlib, all on the same axes. The shaded region is the standard
    deviation of the density, which is calculated using the cross-validation
    folds.
    """
    std_dev = calculate_density_sd(x, by=by, cv_fold=None, grid_bins=grid_bins)
