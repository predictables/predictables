import matplotlib.pyplot as plt
import numpy as np
import warnings


def create_biplot(
    pca,
    feature_names,
    ax=None,
    loading_threshold=0.2,
    figsize=(10, 10),
    backend="matplotlib",
    use_limits=True,
    scatter_alpha=0.5,
    axes_color="k",
    axes_lw=2,
    axes_alpha=0.5,
    axes_ls="--",
    arrow_lw=0.5,
    arrow_alpha=0.5,
    arrow_color="r",
):
    """
    Creates a biplot for the given fitted PCA object.

    Parameters
    ----------
    pca : PCA object
        The fitted PCA object from scikit-learn.
    feature_names : list of str
        The names of the features in the fitted PCA object.
    ax : matplotlib.axes.Axes, optional
        A Matplotlib Axes object to draw the biplot on.
        If not provided, a new figure and axes will be created.
    loading_threshold : float, optional
        The threshold for the absolute value of the loading to be annotated.
        By default, it's set to 0.2. Set to 0 to annotate all loadings.
    figsize : tuple of float, optional
        The figure size, by default (10, 10). Note that this parameter is only
        used if `ax` is not provided. Also consider a square-sized figure, given
        that the axes are percent-scale loading vectors.
    backend : str, optional
        The plotting backend to use. By default, it's set to "matplotlib".
        Currently, only "matplotlib" is supported, but "plotly" support is planned.
    use_limits : bool, optional
        Whether to set the limits of the axes to [-1, 1], by default True.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The Axes object with the biplot.
    """
    if backend == "plotly":
        raise NotImplementedError("Plotly support is not yet implemented")
    elif backend != "matplotlib":
        raise ValueError(f"Backend {backend} is not supported")
    else:
        # Catch UserWarning from matplotlib
        warnings.filterwarnings("ignore")

        if ax is None:
            _, ax = plt.subplots(figsize=figsize)

        # Scores (PCA transformed data)
        scores = pca.transform(pca.components_)

        # Plot scores
        ax.scatter(scores[:, 0], scores[:, 1], alpha=scatter_alpha)

        # Plot X and Y axes
        ax.axhline(0, color=axes_color, linestyle=axes_ls, alpha=axes_alpha, lw=axes_lw)
        ax.axvline(0, color=axes_color, linestyle=axes_ls, alpha=axes_alpha, lw=axes_lw)

        # Plot loadings and feature labels
        for i in range(pca.components_.shape[0]):
            loading_vector = pca.components_[i, :2]
            loading_vector_magnitude = np.sqrt(np.sum(loading_vector**2))

            # Only plot the loading vector if its magnitude exceeds the threshold
            if loading_vector_magnitude > loading_threshold:
                ax.arrow(
                    0,
                    0,
                    loading_vector[0],
                    loading_vector[1],
                    color=arrow_color,
                    alpha=arrow_alpha,
                    lw=arrow_lw,
                    # arrowprops=dict(arrowstyle="->"),
                )
                # shift the text a bit away from the arrow, but in the direction of the arrow
                magnitude = np.sqrt(np.sum(loading_vector**2))
                text_shift = (
                    (magnitude / 10) * loading_vector / np.linalg.norm(loading_vector)
                )
                ax.text(
                    loading_vector[0] + text_shift[0],
                    loading_vector[1] + text_shift[1],
                    f"{feature_names[i]}\n{loading_vector_magnitude:.2f}",
                    color="g",
                    ha="center",
                    va="center",
                    bbox=dict(
                        facecolor="w",
                        edgecolor="k",
                        lw=0.5,
                        alpha=0.6,
                        boxstyle="round,pad=0.2",
                    ),
                )

                # Plot loadings and feature labels
                ax.arrow(
                    0,
                    0,  # Start the arrow at the origin
                    pca.components_[i, 0],
                    pca.components_[i, 1],  # End the arrow at the (PC1, PC2) location
                    color=arrow_color,
                    alpha=arrow_alpha,
                    lw=arrow_lw,
                    # arrowprops=dict(arrowstyle="->"),
                )

        # Set limits
        if use_limits:
            ax.set_xlim([-1, 1])
            ax.set_ylim([-1, 1])

        # Add labels and title
        ax.set_xlabel(
            f"First Principal Component - {100*pca.explained_variance_ratio_[0]:.1f}% Of the Total Variance Explained"
        )
        ax.set_ylabel(
            f"Second Principal Component - {100*pca.explained_variance_ratio_[1]:.1f}% Of the Total Variance Explained"
        )
        ax.set_title(
            f"PCA Biplot - {100*(pca.explained_variance_ratio_[0] + pca.explained_variance_ratio_[1]):.1f}% Variance Explained by First Two Principal Components"
        )
        ax.grid(True)

        return ax
