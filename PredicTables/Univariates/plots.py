from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.stats import ttest_ind

from PredicTables.util.stats import gini_coefficient, kl_divergence


def _plot_label(s: str) -> str:
    s = s.replace("_", " ").title()
    return f"[{s}]" if s[0] == "[" else s


def get_rc_params() -> dict:
    """
    Returns a dictionary of Matplotlib RC parameters for customizing plot styles.

    The function returns a dictionary of Matplotlib RC parameters that can be used to
    customize the style of Matplotlib plots. The parameters include font sizes, tick
    label sizes, legend font size, figure size, and figure DPI.

    Returns:
    --------
    dict
        A dictionary of Matplotlib RC parameters.
    """
    new_rc = {}

    new_rc["font.size"] = 12
    new_rc["axes.titlesize"] = 16
    new_rc["axes.labelsize"] = 14
    new_rc["xtick.labelsize"] = 14
    new_rc["ytick.labelsize"] = 14
    # new_rc['legend.fontsize'] = 14
    new_rc["figure.titlesize"] = 16

    # Set default figure size
    new_rc["figure.figsize"] = (17, 17)

    # Set default figure dpi
    new_rc["figure.dpi"] = 150

    return new_rc


def set_rc_params(rcParams) -> dict:
    """
    Sets Matplotlib RC parameters for customizing plot styles.

    The function sets Matplotlib RC parameters for customizing the style of Matplotlib
    plots. The parameters include font sizes, tick label sizes, legend font size,
    figure size, and figure DPI. The parameters are obtained from the `get_rc_params`
    function.

    Parameters:
    -----------
    rcParams : dict
        A dictionary of Matplotlib RC parameters to be updated.

    Returns:
    --------
    dict
        A dictionary of Matplotlib RC parameters with the updated values.
    """
    for k, v in get_rc_params().items():
        rcParams[k] = v
    return rcParams


def _rotate_x_labels_if_overlap(ax: plt.Axes) -> plt.Axes:
    """
    Rotates the x-axis tick labels of a given Matplotlib axis if they overlap.

    The function checks if any of the tick labels overlap with each other, and if so,
    rotates them by 10 degrees until there is no overlap. The tick labels are also
    aligned to the right to prevent them from overlapping with the tick marks.

    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        The Matplotlib axis object to rotate the x-axis tick labels of.

    Returns:
    --------
    matplotlib.axes.Axes
        The same Matplotlib axis object with the rotated x-axis tick labels.
    """
    # Draw canvas to populate tick labels, to get their dimensions
    ax.figure.canvas.draw()

    overlap = True
    rotation_angle = 0
    alignment_set = False

    # Keep rotating the tick labels by 10 degrees until there is no overlap
    while overlap and rotation_angle <= 90:
        overlap = False
        for i, label in enumerate(ax.xaxis.get_ticklabels()):
            # Get bounding box of current tick label
            bbox_i = label.get_window_extent()

            for j, label_next in enumerate(ax.xaxis.get_ticklabels()):
                if i >= j:
                    continue

                # Get bounding box of next tick label
                bbox_j = label_next.get_window_extent()

                # Check for overlap between current and next tick labels
                if bbox_i.overlaps(bbox_j):
                    overlap = True
                    break
            if overlap:
                # Align tick labels to the right to prevent overlap with tick marks
                if not alignment_set:
                    for label in ax.xaxis.get_ticklabels():
                        label.set_horizontalalignment("right")
                    alignment_set = True

                # Rotate tick labels by 10 degrees
                rotation_angle += 10
                for label in ax.xaxis.get_ticklabels():
                    label.set_rotation(rotation_angle)
                ax.figure.canvas.draw()
                break

    return ax


def _quintile_lift_plot(
    feature: pd.Series,
    observed_target: pd.Series,
    modeled_target: pd.Series,
    ax: plt.Axes = None,
    modeled_color: str = "red",
    observed_color: str = "lightgreen",
):
    """
    Plots the quintile lift for a given feature and target.

    The function calculates the mean observed target and modeled target for each
    quintile of the modeled target, and plots them as a bar chart. The function
    also calculates the KL divergence and Gini coefficient between the observed
    and modeled targets, and adds them as annotations to the plot.

    Parameters:
    -----------
    feature : pd.Series
        A Pandas Series containing the feature data.
    observed_target : pd.Series
        A Pandas Series containing the observed target data.
    modeled_target : pd.Series
        A Pandas Series containing the modeled target data.
    ax : matplotlib.axes.Axes, optional
        The Matplotlib axis object to plot the quintile lift on. If not provided,
        a new figure and axis object will be created.
    modeled_color : str, optional
        The color of the modeled target bars. Default is 'red'.
    observed_color : str, optional
        The color of the observed target bars. Default is 'lightgreen'.

    Returns:
    --------
    matplotlib.axes.Axes
        The Matplotlib axis object with the quintile lift plot.
    """
    # Create DataFrame to hold all the data
    df = pd.DataFrame(
        {
            "feature": feature,
            "observed_target": observed_target,
            "modeled_target": modeled_target,
        }
    )

    # Create quintile bins based on the modeled target
    # If there are n < 5 unique values, don't bin quintiles -- instead, bin into
    # n bins based on the modeled target
    if len(df["modeled_target"].unique()) < 5:
        df["quintile"] = (
            pd.qcut(
                df["modeled_target"],
                len(df["modeled_target"].unique()),
                labels=False,
                duplicates="drop",
            )
            + 1
        )
    else:
        df["quintile"] = (
            pd.qcut(df["modeled_target"], 5, labels=False, duplicates="drop") + 1
        )

    # Calculate the mean target and modeled target for each quintile
    lift_df = (
        df.groupby("quintile")
        .agg(
            observed_target_mean=("observed_target", "mean"),
            modeled_target_mean=("modeled_target", "mean"),
        )
        .reset_index()
    )

    # Plotting
    if ax is None:
        _, ax = plt.subplots()

    bars1 = ax.bar(
        lift_df["quintile"] - 0.2,
        lift_df["observed_target_mean"],
        0.4,
        label="Observed",
        color=observed_color,
        edgecolor="black",
        alpha=0.5,
    )
    bars2 = ax.bar(
        lift_df["quintile"] + 0.2,
        lift_df["modeled_target_mean"],
        0.4,
        label="Modeled",
        color=modeled_color,
        edgecolor="black",
        alpha=0.5,
    )

    # Add data labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height:.2f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, -3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=16,
                bbox=dict(
                    boxstyle="round,pad=0.25",
                    edgecolor="black",
                    facecolor="white",
                    alpha=0.9,
                ),
            )

    ax.set_xticks(lift_df["quintile"])
    ax.set_xlabel("Modeled Quintile")
    ax.set_ylabel("Mean Target")
    ax.legend()

    # KL Divergence calculation
    kl_div = kl_divergence(
        lift_df["observed_target_mean"].values, lift_df["modeled_target_mean"].values
    )

    # Gini calculation
    gini_coeff = gini_coefficient(df["observed_target"], df["modeled_target"])

    # Add KL divergence and Gini coefficient as annotation to the plot
    ax.annotate(
        f"KL Divergence: {kl_div:.3f}\nGini Coefficient: {gini_coeff:.3f}",
        xy=(0.75, 0.05),
        xycoords="axes fraction",
        fontsize=16,
        ha="center",
        bbox=dict(
            boxstyle="round,pad=0.25", edgecolor="black", facecolor="white", alpha=0.95
        ),
    )

    # plt.tight_layout()
    ax.set_title("Qunitile Lift Plot")

    ax = _rotate_x_labels_if_overlap(ax)
    ax.figure.tight_layout()

    return ax


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


def _calculate_density_sd(densities: np.ndarray) -> np.ndarray:
    """
    Calculate the standard deviation of densities.

    Parameters:
    ----------
    densities: A numpy array of densities.

    Returns:
    --------
    A numpy array of standard deviations of densities.

    Raises:
    -------
    TypeError: If densities is not a numpy array.
    ValueError: If densities is empty or has only one element.

    """
    if not isinstance(densities, np.ndarray):
        raise TypeError("densities must be a numpy array")
    if len(densities) < 2:
        raise ValueError("densities must have at least two elements")

    sd = np.std(densities, axis=0)
    assert isinstance(sd, np.ndarray), "sd must be a numpy array"
    assert len(sd) == len(densities[0]), "sd must have the same length as densities[0]"

    return sd


def _add_shaded_sd(ax, x, density, sd, color, target_val):
    """
    Adds a shaded region to the given axis representing the standard deviation of the
    data.

    Parameters:
    -----------
    ax (matplotlib.axes.Axes): The axis to add the shaded region to.
    density (numpy.ndarray): The density of the data.
    sd (numpy.ndarray): The standard deviation of the data.

    Returns:
    --------
    None
    """
    # Calculate the lower and upper bounds of the shaded region
    lower_bound = pd.Series(density) - pd.Series(sd)
    upper_bound = pd.Series(density) + pd.Series(sd)

    ax.fill_between(
        x,
        lower_bound,
        upper_bound,
        alpha=0.2,
        color=color,
        zorder=-1,
        label=f"Density for {target_val} +/- 1 SD",
    )
    ax.plot(x, lower_bound, color=color, linewidth=0.5, linestyle="dotted", alpha=0.7)
    ax.plot(x, upper_bound, color=color, linewidth=0.5, linestyle="dotted", alpha=0.7)

    # Return the axis
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
    median0, median1 = feature[target == 0].median(), feature[target == 1].median()

    # Add vertical lines
    ax.axvline(mean0, color="blue", linestyle="--", linewidth=1)
    ax.axvline(mean1, color="orange", linestyle="--", linewidth=1)
    ax.axvline(median0, color="blue", linestyle="dotted", linewidth=1)
    ax.axvline(median1, color="orange", linestyle="dotted", linewidth=1)

    # Define annotation position and arrow properties based on mean0 and mean1
    pos0, pos1 = ("right", "left") if mean0 < mean1 else ("left", "right")
    arrowprops0 = dict(arrowstyle="->", lw=1)
    arrowprops1 = dict(arrowstyle="->", lw=1)

    # Annotate for target=0
    ax.annotate(
        f"{target.name}=0\n===========\nMean / Median =\n{mean0 / median0:.2f}",
        xy=(mean0, 0.2),
        xycoords="data",
        xytext=(-20 if pos0 == "right" else 20, -20),
        textcoords="offset points",
        ha=pos0,
        va="bottom",
        fontsize=16,
        bbox=dict(
            boxstyle="round,pad=0.3", edgecolor="black", facecolor="blue", alpha=0.2
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
        fontsize=16,
        bbox=dict(
            boxstyle="round,pad=0.3", edgecolor="black", facecolor="orange", alpha=0.2
        ),
        arrowprops=arrowprops1,
    )

    return ax


def _annotate_ttest_means(
    ax: plt.Axes, feature: pd.Series, target: pd.Series
) -> plt.Axes:
    """
    Annotates the t-test results comparing the means of the feature
    variable for each target class.

    Parameters:
    -----------
    ax (matplotlib.axes.Axes): The axis to add the annotations to.
    feature (pandas.Series): The feature variable data.
    target (pandas.Series): The target variable data.

    Returns:
    --------
    ax (matplotlib.axes.Axes): The axis with the annotations added.
    """

    # Conduct the t-test
    t_stat, p_val = ttest_ind(
        feature[target == 0], feature[target == 1], equal_var=False
    )

    # Prepare the text for the annotation
    ttest_text = f"Results of a t-test:\n============\nt-statistic: {t_stat:.3f}\n\
p-value: {p_val:.3f}"

    # Interpret the p-value
    if p_val < 0.01:
        p_interpret = "Extremely likely to be from\ndifferent distributions"
    elif p_val < 0.05:
        p_interpret = "Likely to be from\ndifferent distributions"
    else:
        p_interpret = "Unlikely to be from\ndifferent distributions"

    # Update the text for the annotation
    ttest_text += f"\n\n{p_interpret}"

    # Add the annotation box
    ax.annotate(
        ttest_text,
        xy=(0.15, 0.8),
        xycoords="axes fraction",
        xytext=(20, 20),
        textcoords="offset points",
        ha="center",
        va="center",
        fontsize=16,
        bbox=dict(
            boxstyle="round,pad=0.3", edgecolor="black", facecolor="white", alpha=0.5
        ),
    )

    return ax
