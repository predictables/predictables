from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.stats import ttest_ind

from PredicTables.util.stats import gini_coefficient, kl_divergence


def _plot_lift_chart(df, feature, target, ax=None, alpha=0.5):
    # Calculate overall positive rate
    overall_positive_rate = df[target].mean()

    # Group by the feature and calculate the mean target variable
    lift_data = df.groupby(feature)[target].mean().reset_index()
    lift_data["lift"] = lift_data[target] / overall_positive_rate

    # Sort by the lift
    lift_data = lift_data.sort_values("lift", ascending=False).reset_index(drop=True)

    # Plotting
    if ax is None:
        _, ax = plt.subplots()

    # Conditionally set color based on lift value
    colors = ["green" if lift > 1 else "red" for lift in lift_data["lift"]]

    ax.bar(
        lift_data[feature],
        lift_data["lift"],
        color=colors,
        alpha=alpha,
        label=lift_data[feature],
    )
    ax.axhline(1, color="black", linestyle="--")
    # ax.set_xlabel(feature)
    ax.set_ylabel("Lift")
    ax.set_title("Lift Chart for " + feature)

    for i, v in enumerate(lift_data["lift"]):
        ax.annotate(
            f"{v:.2f}",
            (i, v),
            xytext=(0, -10),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=16,
            bbox=dict(boxstyle="round,pad=0.3", edgecolor="gray", facecolor="white"),
            zorder=5,
        )

    ax = _rotate_x_labels_if_overlap(ax)
    ax.figure.tight_layout()
    return ax


def plot_violin_with_outliers(
    target: pd.Series,
    feature: pd.Series,
    outlier_df: pd.DataFrame,
    cv_folds_data: List[Tuple[np.ndarray, np.ndarray]],
    n_folds: int = 10,
    cv_alpha: float = 0.2,
    ax: Optional[plt.Axes] = None,
    dropvals: Optional[List] = None,
) -> plt.Axes:
    """
    Plots a violin plot with outliers annotated.

    The function takes a target variable, a feature variable, a DataFrame of outlier
    data, and cross-validation fold data as input. It plots a violin plot of the
    feature variable with the target variable on the y-axis, and annotates the
    outliers in the plot. The function also overlays the cross-validation fold data
    on the plot as a shaded area.

    Parameters:
    -----------
    target : pd.Series
        A Pandas Series containing the target variable data.
    feature : pd.Series
        A Pandas Series containing the feature variable data.
    outlier_df : pd.DataFrame
        A Pandas DataFrame containing the outlier data. The DataFrame should have
        columns 'feature', 'target', and 'is_outlier'.
    cv_folds_data : List[Tuple[np.ndarray, np.ndarray]]
        A list of tuples containing the indices of the training and validation data
        for each cross-validation fold.
    n_folds : int, optional
        The number of cross-validation folds. Default is 10.
    cv_alpha : float, optional
        The alpha value for the cross-validation fold shading. Default is 0.2.
    ax : matplotlib.axes.Axes, optional
        The Matplotlib axis object to plot the violin plot on. If not provided, a new
        figure and axis object will be created.
    dropvals : list, optional
        A list of values to drop from the feature variable data before plotting.
        Default is None.

    Returns:
    --------
    matplotlib.axes.Axes
        The Matplotlib axis object with the violin plot and outliers annotated.
    """
    # Create a new figure and axis object if none are provided
    if ax is None:
        _, ax = plt.subplots()

    # Drop specified values from the feature and target variable data
    if dropvals is not None:
        target = target[~feature.isin(dropvals)]
        feature = feature[~feature.isin(dropvals)]
        outlier_df = outlier_df.loc[~outlier_df[feature.name].isin(dropvals)]

    # Initialize lists to store densities
    densities_0, densities_1 = [], []
    x_vals_0, x_vals_1 = [], []

    # Split the feature variable data by binary target
    filter0, filter1 = (target == 0).values, (target == 1).values

    # Loop over cross-validation folds and plot densities
    added_label = False
    for fold in range(n_folds):
        _, val_idx = cv_folds_data.train[fold], cv_folds_data.val[fold]

        # Create filters for cross-validation for each class
        filter0_cv = np.isin(np.arange(len(feature)), val_idx) & filter0
        filter1_cv = np.isin(np.arange(len(feature)), val_idx) & filter1

        # Filter down to just include the validation-set data
        feature0 = feature[
            np.isin(np.arange(len(feature)), val_idx) & filter0 & filter0_cv
        ]
        feature1 = feature[
            np.isin(np.arange(len(feature)), val_idx) & filter1 & filter1_cv
        ]

        # Plot the density of the feature variable for the current fold and target class
        if added_label:
            density_0, x_val_0 = _plot_violin(
                ax=ax,
                data=feature0,
                side="left",
                orientation="horizontal",
                alpha=cv_alpha,
                dropvals=[-0.01, -1],
            )
            density_1, x_val_1 = _plot_violin(
                ax=ax,
                data=feature1,
                side="right",
                orientation="horizontal",
                alpha=cv_alpha,
                dropvals=[-0.01, -1],
            )
        else:
            density_0, x_val_0 = _plot_violin(
                ax=ax,
                data=feature0,
                side="left",
                orientation="horizontal",
                alpha=cv_alpha,
                dropvals=[-0.01, -1],
                label=f"CV Folds for {target.name} = 0",
            )
            density_1, x_val_1 = _plot_violin(
                ax=ax,
                data=feature1,
                side="right",
                orientation="horizontal",
                alpha=cv_alpha,
                dropvals=[-0.01, -1],
                label=f"CV Folds for {target.name} = 1",
            )
            added_label = True

        # Append the density and x_vals to the lists
        densities_0.append(density_0)
        densities_1.append(density_1)
        x_vals_0.append(x_val_0)
        x_vals_1.append(x_val_1)

    # Calculate the standard deviation of the densities for each target class
    densities_0, densities_1 = np.array(densities_0), np.array(densities_1)
    # return densities_0, densities_1
    sd_0, sd_1 = _calculate_density_sd(densities_0), _calculate_density_sd(densities_1)

    # Plot the density of the feature variable for the full data set for each
    # target class
    density0, x0 = _plot_violin(ax, feature[target == 0], "left", 1.0, linestyle="--")
    density1, x1 = _plot_violin(ax, feature[target == 1], "right", 1.0, linestyle="--")

    # Add Mean/Median annotations
    ax = _annotate_mean_median(ax, feature, target)

    # Add t-test annotations
    ax = _annotate_ttest_means(ax, feature, target)

    # Add shaded regions +/- 1 standard deviation for the density of the feature
    # variable for each target class
    ax = _add_shaded_sd(ax, x0, density0, sd_0, "blue", 0)
    ax = _add_shaded_sd(ax, x1, density1, sd_1, "orange", 1)

    # # Annotate the outliers in the plot for each target class
    # _annotate_outliers(ax, feature[target == 0], outlier_df, 'left')
    # _annotate_outliers(ax, feature[target == 1], outlier_df, 'right')

    # Set the title of the plot
    ax.set_title(
        f"Density Plot of [{_plot_label(feature.name)}] by \
[{_plot_label(target.name)}]"
    )

    ax.legend()

    return ax  # , density0, density1, sd_0, sd_1


def _plot_violin(
    ax: plt.Axes,
    data: pd.Series,
    side: str,
    orientation: str = "vertical",
    alpha: float = 1.0,
    fill: bool = False,
    linestyle: str = "-",
    dropvals: list = None,
    **kwargs: dict,
) -> plt.Axes:
    """
    Plots a single-sided violin plot of a Pandas Series.

    The function takes a Matplotlib axis object, a Pandas Series of data, a side
    ('left' or 'right'), an orientation ('vertical' or 'horizontal'), an alpha value,
    a fill flag, and a list of values to drop from the data before plotting. It plots
    a single-sided violin plot of the data on the specified side of the axis object,
    with the specified orientation. The function computes a kernel density estimate of
    the data, and smooths the estimate using a convolution filter. The function then
    plots the smoothed density estimate on the specified side of the axis object.

    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        The Matplotlib axis object to plot the violin plot on.
    data : pd.Series
        A Pandas Series of data to plot.
    side : str
        The side of the axis object to plot the violin plot on ('left' or 'right').
    orientation : str, optional
        The orientation of the violin plot ('vertical' or 'horizontal').
        Default is 'vertical'.
    alpha : float, optional
        The alpha value for the plot. Default is 1.0.
    fill : bool, optional
        A flag indicating whether to fill the area under the density curve.
        Default is False.
    linestyle : str, optional
        The line style for the plot. Default is '-'.
    dropvals : list, optional
        A list of values to drop from the data before plotting. Default is None.
    **kwargs : dict, optional
        Additional keyword arguments to pass to the Matplotlib plot function.

    Returns:
    --------
    matplotlib.axes.Axes
        The Matplotlib axis object with the violin plot plotted on the specified side.
    """
    # Drop values if specified
    if dropvals is not None:
        data = data[~data.isin(dropvals)]

    # Compute the kernel density estimate with updated data
    n = len(data)
    min_bins = 10
    max_bins = 50

    # Number of bins bounded by min/max bins
    n_bins = min(max(int(np.sqrt(n)), min_bins), max_bins)

    # Compute the histogram
    try:
        bin_edges = np.histogram_bin_edges(data, bins=n_bins)
        hist, _ = np.histogram(data, bins=bin_edges, density=True)
    except ValueError:
        hist, bin_edges = np.histogram(data, bins="auto", density=True)

    # Smooth the histogram
    # original_ave = [0.2, 0.6, 0.2]
    # middle_ave = [0.1, 0.2, 0.4, 0.2, 0.1]
    middle_peak = [0.05, 0.1, 0.15, 0.4, 0.15, 0.1, 0.05]
    # wider_ave = [0.05, 0.1, 0.2, 0.3, 0.2, 0.1, 0.05]
    hist_smooth = np.convolve(hist, middle_peak, mode="same")

    # Create the x-values
    x_vals = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Apply cubic spline interpolation
    f = interp1d(x_vals, hist_smooth, kind="cubic")
    new_x = np.linspace(min(x_vals), max(x_vals), 100)
    new_hist_smooth = f(new_x)

    # Mirror the density values for plotting on one side
    plot_x_vals = -new_x if (side == "left" and orientation == "vertical") else new_x

    # Plot the density
    if orientation == "vertical":
        ax.plot(
            new_hist_smooth,
            plot_x_vals,
            alpha=alpha,
            color=("blue" if side == "left" else "orange"),
            linestyle=linestyle,
            **kwargs,
        )
        if fill:
            ax.fill_betweenx(
                plot_x_vals,
                new_hist_smooth,
                alpha=alpha,
                color=("blue" if side == "left" else "orange"),
            )
    else:
        ax.plot(
            plot_x_vals,
            new_hist_smooth,
            alpha=alpha,
            color=("blue" if side == "left" else "orange"),
            linestyle=linestyle,
            **kwargs,
        )
        if fill:
            ax.fill_between(
                plot_x_vals,
                new_hist_smooth,
                alpha=alpha,
                color=("blue" if side == "left" else "orange"),
            )

    return new_hist_smooth, new_x



