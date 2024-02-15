from typing import List, Optional, Tuple  # type: ignore

import matplotlib.pyplot as plt  # type: ignore
import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import plotly.graph_objects as go  # type: ignore
from matplotlib.axes import Axes  # type: ignore
from sklearn.decomposition import PCA  # type: ignore

# def create_loading_plot(
#     pca,
#     feature_names,
#     n_components: int = 10,
#     ax: Optional[Axes] = None,
#     fig: Optional[go.figure] = None,
#     average_loading_threshold: float = 0.05,
#     max_features: Optional[int] = None,
#     figsize: Tuple[float, float] = (10, 10),
#     backend: str = "matplotlib",
#     bar_alpha: float = 0.8,
#     bar_width: float = 0.9,
#     main_title_fontsize: float = 13,
#     main_title_fontweight: str = "bold",
#     sub_title_fontsize: float = 10,
#     legend_fontsize: float = 9,
#     x_ticks_rotation: float = 45,
#     x_label_fontsize: float = 10,
#     y_label_fontsize: float = 10,
#     include_legend: bool = True,
#     drop_legend_when_n_features: float = 15,
# ):
#     # `n_components` must be less than or equal to the number of components in the PCA
#     n_components = min(n_components, pca.n_components_)

#     # cumulative loading threshold is the average loading threshold times the number
# of components
#     cumulative_loading_threshold = average_loading_threshold * n_components

#     # Decide if there will be a legend
#     if max_features is None:
#         max_features = feature_names.shape[0]

#     # Override the legend if there are too many features
#     if max_features >= drop_legend_when_n_features:
#         include_legend = False

#         # Get the loadings
#     loadings = pca.components_.T
#     loadings = np.array([x[:n_components] for x in loadings])

#     # Create a dataframe with the loadings
#     df = pd.DataFrame(
#         loadings,
#         columns=[f"PC-{'0' if i < 9 else ''}{i+1}" for i in range(n_components)],
#         index=feature_names,
#     )

#     # Absolute value of the loadings to tighten the plot
#     df = df.abs()

#     # Get the number of features
#     total_features = df.shape[0]
#     filtered_features = False

#     # Sort the loadings by the sum of the first `n_components` columns
#     df["sort_col"] = df.cumsum(axis=1).iloc[:, n_components - 1]

#     # Test that sort_col is a date or datetime column
#     df = df.sort_values(by="sort_col", ascending=False)
#     df = df.loc[df.sort_col > cumulative_loading_threshold, :]
#     df = df.drop(columns=["sort_col"])
#     hidden_features = total_features - df.shape[0]
#     if hidden_features > 0:
#         filtered_features = True
#         filter_type = "loading_threshold"

#     # Only show the top `max_features` features, or all features if there are less
# than `max_features`
#     if max_features is None:
#         max_features = total_features
#     elif df.shape[0] > max_features:
#         df = df.iloc[:max_features, :]
#         filtered_features = True
#         filter_type = "max_features"
#         hidden_features = total_features - max_features

#     # Get the explained variance for the first `n_components` components (for
# the title)
#     explained_variance = pca.explained_variance_ratio_[:n_components].sum()

#     # Test to show if any features were hidden
#     hidden_features_text = (
#         f" - {hidden_features} features are hidden" if hidden_features > 0 else ""
#     )

#     # Labels and other text elements will be the same regardless of backend
#     x_label_no_filter = "Features" + (
#         f" (Showing all with an ave loading for each of the {n_components}
# components > {average_loading_threshold:.2f}{hidden_features_text})"
#         if filtered_features
#         else ""
#     )
#     x_label_filter = "Features" + (
#         (
#             f" (Only showing the top {max_features} features{hidden_features_text})"
#             if filter_type == "max_features"
#             else f" (Showing all with an ave loading for each of the {n_components}
# components > {average_loading_threshold:.2f}{hidden_features_text})"
#         )
#         if filtered_features
#         else ""
#     )

#     # If the backend is plotly, raise a not implemented error
#     if backend == "plotly":
#         raise NotImplementedError("Plotly support is not yet implemented")
#     elif backend != "matplotlib":
#         raise ValueError(f"Backend {backend} is not supported")
#     else:
#         # Create a new figure and axes if needed
#         if ax is None:
#             _, ax = plt.subplots(figsize=figsize)

#         # Plot the loadings for the first `n_components` components as a stacked
# bar plot
#         if include_legend:
#             df.iloc[:, :n_components].plot.bar(
#                 stacked=True,
#                 ax=ax,
#                 width=bar_width,
#                 alpha=bar_alpha,
#                 color=[f"C{i}" for i in range(n_components)],
#                 label=[f"PC-{i+1}" for i in range(n_components)],
#             )
#         else:
#             df.iloc[:, :n_components].plot.bar(
#                 stacked=True,
#                 ax=ax,
#                 width=bar_width,
#                 alpha=bar_alpha,
#                 color=[f"C{i}" for i in range(n_components)],
#             )

#         ax.set_xlabel(
#             x_label_filter if filtered_features else x_label_no_filter,
#             fontsize=x_label_fontsize,
#         )

#         main_title_text = "Cumulative Influence On Explained Variance"
#         y_label_text = "Cumulative Absolute Loading"
#         ax.set_ylabel(y_label_text, fontsize=y_label_fontsize)
#         plt.suptitle(
#             main_title_text,
#             fontsize=main_title_fontsize,
#             fontweight=main_title_fontweight,
#         )

#         sub_title_text = f"The cumulative absolute value of the loadings for each
# feature for the first {n_components} principal components\nThis plot indicates the
# features' relative contributions to the {explained_variance:.1%} of variance explained
# by the {n_components} components"
#         ax.set_title(
#             sub_title_text,
#             fontsize=sub_title_fontsize,
#         )
#         plt.xticks(rotation=x_ticks_rotation, ha="right")

#         # Add a legend if needed
#         if filtered_features:
#             if include_legend:
#                 plt.legend(
#                     fontsize=legend_fontsize,
#                     bbox_to_anchor=(1.05, 1),
#                     loc="upper left",
#                 )
#             else:
#                 ax.get_legend().remove()
#         return ax


def get_loadings(pca: PCA, feature_names: List[str], n_components: int) -> pd.DataFrame:
    """
    Get the loadings for the first `n_components` components

    Parameters
    ----------
    pca : PCA
        The PCA model
    feature_names : List[str]
        The feature names
    n_components : int
        The number of components to get the loadings for

    Returns
    -------
    pd.DataFrame
        The loadings for the first `n_components` components
    """
    n_components = min(n_components, pca.n_components_)
    loadings = pca.components_.T
    loadings = np.array([x[:n_components] for x in loadings])
    df = pd.DataFrame(
        loadings,
        columns=[f"PC-{'0' if i < 9 else ''}{i+1}" for i in range(n_components)],
        index=feature_names,
    )
    df = df.abs()
    return df


def filter_loadings(
    df: pd.DataFrame,
    n_components: int,
    average_loading_threshold: float,
    max_features: Optional[int],
) -> Tuple[pd.DataFrame, bool, str, int]:
    """
    Filter the loadings based on the average loading threshold and the maximum number
    of features

    Parameters
    ----------
    df : pd.DataFrame
        The loadings
    n_components : int
        The number of components
    average_loading_threshold : float
        The average loading threshold
    max_features : Optional[int]
        The maximum number of features to show

    Returns
    -------
    Tuple[pd.DataFrame, bool, str, int]
        The filtered loadings, a boolean indicating if features were filtered, the
        filter type, and the number of hidden features
    """
    # Step 1: Sort the loadings by the sum of the first `n_components` columns
    total_features = df.shape[0]
    filtered_features = False
    df["sort_col"] = df.cumsum(axis=1).iloc[:, n_components - 1]
    df = df.sort_values(by="sort_col", ascending=False)

    # Step 2: Filter the loadings based on the average loading threshold
    cumulative_loading_threshold = average_loading_threshold * n_components
    df = df.loc[df.sort_col > cumulative_loading_threshold, :]
    df = df.drop(columns=["sort_col"])
    hidden_features = total_features - df.shape[0]
    if hidden_features > 0:
        filtered_features = True
        filter_type = "loading_threshold"
    if max_features is None:
        max_features = total_features
    elif df.shape[0] > max_features:
        df = df.iloc[:max_features, :]
        filtered_features = True
        filter_type = "max_features"
        hidden_features = total_features - max_features
    return df, filtered_features, filter_type, hidden_features


def plot_loadings(df, n_components, ax, bar_width, bar_alpha, include_legend):
    if include_legend:
        df.iloc[:, :n_components].plot.bar(
            stacked=True,
            ax=ax,
            width=bar_width,
            alpha=bar_alpha,
            color=[f"C{i}" for i in range(n_components)],
            label=[f"PC-{i+1}" for i in range(n_components)],
        )
    else:
        df.iloc[:, :n_components].plot.bar(
            stacked=True,
            ax=ax,
            width=bar_width,
            alpha=bar_alpha,
            color=[f"C{i}" for i in range(n_components)],
        )


def _matplotlib_plot(
    df: pd.DataFrame,
    n_components: int = 10,
    bar_width: float = 0.9,
    bar_alpha: float = 0.8,
    include_legend: bool = True,
    ax: Optional[Axes] = None,
    figsize: Tuple[float, float] = (7, 7),
    x_label_fontsize: float = 10,
    y_label_fontsize: float = 10,
    main_title_fontsize: float = 13,
    main_title_fontweight: str = "bold",
    filtered_features: bool = False,
    x_label_no_filter: str = "Features",
    x_label_filter: str = "Features",
    explained_variance: float = 0.0,
    sub_title_fontsize: float = 10,
    x_ticks_rotation: float = 45,
) -> Axes:
    """
    Create a loading plot using matplotlib

    Parameters
    ----------
    df : pd.DataFrame
        The loadings
    n_components : int, optional
        The number of components to show, by default 10
    bar_width : float, optional
        The width of the bars, by default 0.9
    bar_alpha : float, optional
        The transparency of the bars, by default 0.8
    include_legend : bool, optional
        Whether to include a legend, by default True
    ax : Optional[Axes], optional
        The axes to use, by default None
    figsize : Tuple[float, float], optional
        The figure size, by default (7, 7)
    x_label_fontsize : float, optional
        The x label fontsize, by default 10
    y_label_fontsize : float, optional
        The y label fontsize, by default 10
    main_title_fontsize : float, optional
        The main title fontsize, by default 13
    main_title_fontweight : str, optional
        The main title fontweight, by default "bold"
    filtered_features : bool, optional
        Whether the features were filtered, by default False
    x_label_no_filter : str, optional
        The x label when no features were filtered, by default "Features"
    x_label_filter : str, optional
        The x label when features were filtered, by default "Features"
    explained_variance : float, optional
        The explained variance, by default 0.0
    sub_title_fontsize : float, optional
        The subtitle fontsize, by default 10
    x_ticks_rotation : float, optional
        The rotation of the x ticks, by default 45

    Returns
    -------
    Axes
        The axes
    """
    # Create a new figure and axes if needed
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    # Plot the loadings for the first `n_components` components as a stacked bar plot
    plot_loadings(df, n_components, ax, bar_width, bar_alpha, include_legend)

    ax.set_xlabel(
        x_label_filter if filtered_features else x_label_no_filter,
        fontsize=x_label_fontsize,
    )

    main_title_text = "Cumulative Influence On Explained Variance"
    y_label_text = "Cumulative Absolute Loading"
    ax.set_ylabel(y_label_text, fontsize=y_label_fontsize)
    plt.suptitle(
        main_title_text,
        fontsize=main_title_fontsize,
        fontweight=main_title_fontweight,
    )

    sub_title_text = (
        "The cumulative absolute value of the loadings for each feature "
        f"for the first {n_components} principal components\nThis plot indicates the "
        f"features' relative contributions to the {explained_variance:.1%} of variance "
        f"explained by the {n_components} components"
    )
    ax.set_title(
        sub_title_text,
        fontsize=sub_title_fontsize,
    )
    plt.xticks(rotation=x_ticks_rotation, ha="right")

    return ax


def _plotly_plot(*args, **kwargs):
    raise NotImplementedError("Plotly support is not yet implemented")


# trunk-ignore(sourcery/low-code-quality)
def create_loading_plot(
    pca: PCA,
    feature_names: List[str],
    n_components: int = 10,
    ax: Optional[Axes] = None,
    fig: Optional[go.Figure] = None,
    average_loading_threshold: float = 0.05,
    max_features: Optional[int] = None,
    figsize: Tuple[float, float] = (10, 10),
    backend: str = "matplotlib",
    bar_alpha: float = 0.8,
    bar_width: float = 0.9,
    main_title_fontsize: float = 13,
    main_title_fontweight: str = "bold",
    sub_title_fontsize: float = 10,
    legend_fontsize: float = 9,
    x_ticks_rotation: float = 45,
    x_label_fontsize: float = 10,
    y_label_fontsize: float = 10,
    include_legend: bool = True,
    drop_legend_when_n_features: float = 15,
):
    # Get the loadings
    df = get_loadings(pca, feature_names, n_components)

    # Filter the loadings
    df, filtered_features, filter_type, hidden_features = filter_loadings(
        df, n_components, average_loading_threshold, max_features
    )

    # Get the explained variance for the first `n_components` components (for the title)
    explained_variance = pca.explained_variance_ratio_[:n_components].sum()

    # Test to show if any features were hidden
    hidden_features_text = (
        f" - {hidden_features} features are hidden" if hidden_features > 0 else ""
    )

    # Labels and other text elements will be the same regardless of backend
    x_label_no_filter = "Features" + (
        f" (Showing all with an ave loading for each of the {n_components} components "
        f"> {average_loading_threshold:.2f}{hidden_features_text})"
        if filtered_features
        else ""
    )
    x_label_filter = "Features" + (
        (
            f" (Only showing the top {max_features} features{hidden_features_text})"
            if filter_type == "max_features"
            else (
                f" (Showing all with an ave loading for each of the {n_components} "
                f"components > {average_loading_threshold:.2f}{hidden_features_text})"
            )
        )
        if filtered_features
        else ""
    )
    backend_options = {
        "plotly": lambda: _plotly_plot(
            df=df,
            n_components=n_components,
            fig=fig,
            bar_width=bar_width,
            bar_alpha=bar_alpha,
            include_legend=include_legend,
        ),
        "matplotlib": lambda: _matplotlib_plot(
            df=df,
            n_components=n_components,
            bar_width=bar_width,
            bar_alpha=bar_alpha,
            include_legend=include_legend,
            ax=ax,
            figsize=figsize,
            x_label_fontsize=x_label_fontsize,
            y_label_fontsize=y_label_fontsize,
            main_title_fontsize=main_title_fontsize,
            main_title_fontweight=main_title_fontweight,
            filtered_features=filtered_features,
            x_label_no_filter=x_label_no_filter,
            x_label_filter=x_label_filter,
            explained_variance=explained_variance,
            sub_title_fontsize=sub_title_fontsize,
            x_ticks_rotation=x_ticks_rotation,
        ),
    }

    backend_options.get(backend, lambda: None)()

    # If the backend is plotly, raise a not implemented error
    # trunk-ignore(sourcery/switch)
    if backend == "plotly":
        raise NotImplementedError("Plotly support is not yet implemented")
    elif backend != "matplotlib":
        raise ValueError(f"Backend {backend} is not supported")
    else:
        # Create a new figure and axes if needed
        if ax is None:
            _, ax = plt.subplots(figsize=figsize)

        # Plot the loadings for the first `n_components` components as a stacked
        # bar plot
        plot_loadings(df, n_components, ax, bar_width, bar_alpha, include_legend)

        ax.set_xlabel(
            x_label_filter if filtered_features else x_label_no_filter,
            fontsize=x_label_fontsize,
        )

        ax.set_ylabel("Cumulative Absolute Loading", fontsize=y_label_fontsize)

        main_title_text = "Cumulative Influence On Explained Variance"
        plt.suptitle(
            main_title_text,
            fontsize=main_title_fontsize,
            fontweight=main_title_fontweight,
        )

        sub_title_text = (
            "The cumulative absolute value of the loadings for each "
            f"feature for the first {n_components} principal components\n"
            "This plot indicates the features' relative contributions to the "
            f"{explained_variance:.1%} of variance explained by the "
            f"{n_components} components"
        )
        ax.set_title(
            sub_title_text,
            fontsize=sub_title_fontsize,
        )
        if include_legend:
            if filtered_features:
                plt.legend(
                    fontsize=legend_fontsize,
                    bbox_to_anchor=(1.05, 1),
                    loc="upper left",
                )
            else:
                ax.get_legend().remove()
        return ax
