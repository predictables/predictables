import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def density_plot_matplotlib()

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
