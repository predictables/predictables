from typing import List, Tuple, Optional

from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA  # type: ignore

from predictables.util import to_pl_df

from ._preprocessing import preprocess_data_for_pca


def create_scree_plot(
    X: np.ndarray,
    variance_levels: Optional[List[float]] = None,
    y_pos_adjustment: float = 0.1,
    ax: Optional[Axes] = None,
    figsize: Tuple[int, int] = (7, 7),
):
    """
    Creates a scree plot for PCA analysis of the given dataset.

    This function performs Principal Component Analysis (PCA) on the provided dataset
    and creates a scree plot that shows the cumulative variance explained by each
    component. The plot is annotated with lines indicating specified levels of
    cumulative variance explained, aiding in the decision of how many components to
    retain.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        The input dataset for PCA.
    variance_levels : list of float, optional
        A list of levels at which to annotate the cumulative variance.
        For example, [0.75, 0.90, 0.95, 0.99]. By default, it's set to
        [0.75, 0.90, 0.95, 0.99].
    y_pos_adjustment : float, optional
        Adjustment for the y position of the annotations, by default 0.05.
    ax : matplotlib.axes.Axes, optional
        The Axes object to plot on. If None, a new figure and Axes object is created.
    figsize : tuple of int, optional
        The size of the figure, by default (10, 10).

    Returns
    -------
    matplotlib.axes.Axes
        The Axes object with the scree plot.

    Notes
    -----
    The scree plot is a graphical representation of the amount of variance each
    principal component retains. It is useful for determining the number of components
    to include in further analyses or models. The plot includes horizontal lines
    indicating predefined levels of cumulative variance, which are useful for
    selecting a suitable number of components.

    Examples
    --------
    >>> from sklearn.datasets import load_iris
    >>> X, _ = load_iris(return_X_y=True)
    >>> ax = create_scree_plot(X)
    >>> plt.show()

    Raises
    ------
    ValueError
        If `X` is not a 2-dimensional array.
    """
    if len(X.shape) != 2:
        raise ValueError("X must be a 2-dimensional array")

    if variance_levels is None:
        variance_levels = [0.75, 0.90, 0.95, 0.99]

    # Perform PCA
    X0 = to_pl_df(pd.DataFrame(X))
    X_ = preprocess_data_for_pca(X0).to_numpy()
    pca = PCA().fit(X_)
    variance_ratios = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(variance_ratios)

    # Create scree plot
    if ax is None:
        _, ax0 = plt.subplots(figsize=figsize)
    else:
        ax0 = ax
    ax0.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker="o")

    # Get features
    n_features = X.shape[1]

    # Components required for each level
    components = []

    # Add annotations for each variance level
    for level in variance_levels:
        # n_components = np.argmin(np.round(cumulative_variance, 4) >= level) + 1
        n_components = (
            np.sum(np.less_equal(np.round(cumulative_variance, 4), level)) + 1
        )
        y_pos = level - (y_pos_adjustment) * (2 - cumulative_variance[n_components - 1])
        components.append(n_components)

        # Linearly interpolate between points (n_components, level) and
        # (n_components - 1, cumulative_variance[n_components - 2])
        # in the x direction to find the point where the horizontal line
        # intersects with the cumulative variance curve
        def interpolate_x(x1, x2, y1, y2, y):
            return x1 + (x2 - x1) * (y - y1) / (y2 - y1)

        # Horizontal line
        extends_to_x = interpolate_x(
            n_components - 1,
            n_components,
            cumulative_variance[n_components - 2],
            cumulative_variance[n_components - 1],
            level,
        )

        # Do the same for the vertical line
        extends_to_y = np.interp(
            extends_to_x,
            [n_components - 1, n_components],
            [
                cumulative_variance[n_components - 2],
                cumulative_variance[n_components - 1],
            ],
        )

        # Plot the horizontal and vertical lines
        ax0.plot(
            [0, extends_to_x],
            [extends_to_y, extends_to_y],
            color="r",
            linestyle="--",
        )
        ax0.plot(
            [extends_to_x, extends_to_x],
            [0, extends_to_y],
            color="g",
            linestyle="--",
        )

        # Give the data point corresponding to the level a red circle instead of
        # the normal blue circle
        ax0.plot(
            n_components,
            cumulative_variance[n_components - 1],
            marker="o",
            color="r",
            markersize=8,
        )

        # Annotation
        bbox_props = dict(
            boxstyle="round,pad=0.3",  # Rounded box with padding
            ec="black",  # Black edges
            lw=1,  # Thin border
            fc="white",
            alpha=0.75,  # White, partly see-through face
        )
        ax0.annotate(
            f"{int(level*100)}%\n{n_components} "
            f"component{'s' if n_components > 1 else ''}",
            xy=(
                extends_to_x,
                extends_to_y,
            ),  # (x, y) of annotation - what the arrow points to
            xytext=(
                extends_to_x + (n_features / 9),
                y_pos,
            ),  # (x, y) of text - where the text is placed
            textcoords="data",
            ha="center",
            va="center",
            bbox=bbox_props,
            fontweight="bold",
            arrowprops=dict(arrowstyle="->"),
        )

    # One more annotation for the first component, and the first component that
    # explains 100% of the variance
    n_components_for_100 = np.sum(np.less_equal(cumulative_variance, 0.9999)) + 1
    ax0.annotate(
        f"100%\n{n_components_for_100} "
        f"component{'s' if n_components_for_100 > 1 else ''}",
        xy=(n_components_for_100, 1),
        xytext=(n_components_for_100 + 0.75, 0.925),
        textcoords="data",
        ha="center",
        va="center",
        bbox=bbox_props,
        fontweight="bold",
        arrowprops=dict(arrowstyle="->"),
    )

    # Put annotation box in bottom right that explains the scree plot and how to
    # interpret it
    scree_interpretation = (
        "The scree plot shows the cumulative variance "
        "explained by each principal component. I have "
        "added annotations to show the number of components "
        "required to explain at least 75%, 90%, 95%, and 99% "
        "of the variance. The first component that explains"
        "100% of the variance is also annotated. The scree "
        "plot is used to help select the number of components "
        "to retain. In this case, I would likely retain "
        f"{components[variance_levels.index(0.9)]:d}"
        "components to retain at least 90% of the variance."
    )
    if ax0 is not None:
        ax0.text(
            0.975,
            0.025,
            scree_interpretation,
            transform=ax0.transAxes,
            ha="right",
            va="bottom",
            bbox=dict(
                facecolor="w",
                edgecolor="k",
                lw=0.5,
                alpha=0.6,
                boxstyle="round,pad=0.2",
            ),
        )

    # Set limits
    if ax0 is not None:
        ax0.set_xlabel("Number of Components")
        ax0.set_ylabel("Cumulative Explained Variance")
        plt.suptitle("Scree Plot", fontsize=16, fontweight="bold")
        ax0.set_title("Cumulative Variance Explained by N Principal Components")

    return ax0
