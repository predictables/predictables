# # refactored into smaller sub-functions:
# from typing import Any, Optional, Tuple, Union

# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
# from matplotlib.axes import Axes  # type: ignore
# from sklearn.decomposition import PCA  # type: ignore


# def validate_inputs(
#     pca: PCA,
#     feature_names: np.ndarray,
#     n_components: int = 10,
#     max_features: Optional[int] = None,
#     drop_legend_when_n_features: int = 15,
# ) -> Tuple[int, int, bool]:
#     """
#     Validate the inputs to the create_loading_plot function.

#     Parameters
#     ----------
#     pca : sklearn.decomposition.PCA
#         The PCA object to be plotted.
#     feature_names : np.ndarray
#         The names of the features.
#     n_components : int
#         The number of components to plot.
#     max_features : int
#         The maximum number of features to plot.
#     drop_legend_when_n_features : int
#         The number of features to plot before dropping the legend.

#     Returns
#     -------
#     n_components : int
#         The number of components to plot.
#     max_features : int
#         The maximum number of features to plot.
#     include_legend : bool
#         Whether or not to include a legend in the plot.

#     Examples
#     --------
#     >>> from sklearn.decomposition import PCA
#     >>> from predictables.pca.src._create_loading_plot import validate_inputs
#     >>> pca = PCA(n_components=3)
#     >>> feature_names = np.array(['a', 'b', 'c'])


#     """
#     # Validate n_components
#     n_components = min(n_components, pca.n_components_)

#     # Validate max_features and legend inclusion
#     if max_features is None:
#         max_features = feature_names.shape[0]

#     # Determine if legend should be included
#     include_legend = max_features >= drop_legend_when_n_features

#     return n_components, max_features, include_legend


# def calculate_cumulative_loading_threshold(
#     n_components: int = 10, average_loading_threshold: float = 0.05
# ) -> float:
#     """
#     Calculate the cumulative loading threshold.

#     Parameters
#     ----------
#     n_components : int
#         The number of components to plot.
#     average_loading_threshold : float
#         The average loading threshold.

#     Returns
#     -------
#     cumulative_loading_threshold : float
#         The cumulative loading threshold.

#     Examples
#     --------
#     >>> from predictables.pca.src._create_loading_plot import
# calculate_cumulative_loading_threshold
#     >>> n_components = 2
#     >>> average_loading_threshold = 0.05
#     >>> calculate_cumulative_loading_threshold(n_components,
# average_loading_threshold)
#     0.1
#     """
#     return average_loading_threshold * n_components


# def prepare_loadings_data(
#     pca: PCA, feature_names: np.ndarray, n_components: int
# ) -> pd.DataFrame:
#     """
#     Prepare the loadings data for plotting. This function will create a dataframe
#     with the loadings, that has the features as the index and the principal components
#     as the columns.

#     Parameters
#     ----------
#     pca : sklearn.decomposition.PCA
#         The PCA object to be plotted.
#     feature_names : np.ndarray
#         The names of the features.
#     n_components : int
#         The number of components to plot.

#     Returns
#     -------
#     df : pd.DataFrame
#         The dataframe with the loadings.

#     Examples
#     --------
#     >>> from sklearn.decomposition import PCA
#     >>> from predictables.pca.src._create_loading_plot import prepare_loadings_data
#     >>> pca = PCA()
#     >>> n_components = 2
#     >>> feature_names = np.array(['a', 'b', 'c'])
#     >>> df = prepare_loadings_data(pca, feature_names, n_components)
#     >>> df.shape
#     (3, 2)

#     >>> df.columns
#     Index(['PC-01', 'PC-02'], dtype='object')

#     >>> df.index
#     Index(['a', 'b', 'c'], dtype='object')
#     """
#     loadings = pca.components_.T
#     loadings = np.array([x[:n_components] for x in loadings])

#     return pd.DataFrame(
#         loadings,
#         columns=[f"PC-{'0' if i < 9 else ''}{i+1}" for i in range(n_components)],
#         index=feature_names,
#     ).abs()


# def filter_features(df, cumulative_loading_threshold, max_features):
#     total_features = df.shape[0]
#     filtered_features = False
#     hidden_features = 0
#     filter_type = None

#     # Sort and filter by cumulative loading threshold
#     df["sort_col"] = df.cumsum(axis=1).iloc[:, -1]
#     df = df.sort_values(by="sort_col", ascending=False)
#     df = df.loc[df.sort_col > cumulative_loading_threshold, :]
#     df = df.drop(columns=["sort_col"])
#     hidden_features += total_features - df.shape[0]
#     if hidden_features > 0:
#         filtered_features = True
#         filter_type = "loading_threshold"

#     # Further filter by max_features
#     if df.shape[0] > max_features:
#         df = df.iloc[:max_features, :]
#         filtered_features = True
#         filter_type = "max_features"
#         hidden_features = total_features - max_features

#     return df, filtered_features, hidden_features, filter_type


# def create_plot_axes(
#     ax: Optional[Axes] = None, figsize: Tuple[int, int] = (10, 6)
# ) -> Tuple[Optional[Union[Tuple[plt.Figure, Any], plt.Figure]], plt.Axes]:
#     """
#     Create the plot axes.

#     Parameters
#     ----------
#     ax : Optional[Axes]
#         The axes to plot to.
#     figsize : Tuple[int, int]
#         The figure size.

#     Returns
#     -------
#     fig : plt.Figure
#         The figure.
#     ax : plt.Axes
#         The axes.
#     """
#     if ax is None:
#         fig, ax0 = plt.subplots(figsize=figsize)
#     else:
#         fig = ax.get_figure()  # type: ignore # This cannot be None because of the
# conditional
#         ax0 = ax
#     return fig, ax0


# # This function will be large due to the nature of plotting
# def plot_loadings(df, ax, n_components, include_legend, bar_alpha, bar_width):
#     params = dict(
#         stacked=True,
#         ax=ax,
#         width=bar_width,
#         alpha=bar_alpha,
#         color=[f"C{i}" for i in range(n_components)],
#         label=[f"{'' if include_legend else '_'}PC-{i+1}" for i in
# range(n_components)],
#     )
#     return df.iloc[:, :n_components].plot.bar(**params)


# def format_plot(
#     ax,
#     df,
#     n_components,
#     explained_variance,
#     filtered_features,
#     hidden_features,
#     x_label_no_filter,
#     x_label_filter,
#     y_label_text,
#     main_title_text,
#     sub_title_text,
#     main_title_fontsize,
#     main_title_fontweight,
#     sub_title_fontsize,
#     x_ticks_rotation,
#     x_label_fontsize,
#     y_label_fontsize,
#     legend_fontsize,
#     include_legend,
# ):
#     ax.set_xlabel(
#         x_label_filter if filtered_features else x_label_no_filter,
#         fontsize=x_label_fontsize,
#     )
#     ax.set_ylabel(y_label_text, fontsize=y_label_fontsize)

#     plt.suptitle(
#         main_title_text,
#         fontsize=main_title_fontsize,
#         fontweight=main_title_fontweight,
#     )
#     ax.set_title(
#         sub_title_text,
#         fontsize=sub_title_fontsize,
#     )
#     plt.xticks(rotation=x_ticks_rotation, ha="right")

#     if include_legend:
#         plt.legend(
#             fontsize=legend_fontsize,
#             bbox_to_anchor=(1.05, 1),
#             loc="upper left",
#         )
#     else:
#         ax.get_legend().remove()
