import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def create_loading_plot(
    pca,
    feature_names,
    n_components=10,
    ax=None,
    fig=None,
    average_loading_threshold=0.05,
    max_features=None,
    figsize=(10, 10),
    backend="matplotlib",
    bar_alpha=0.8,
    bar_width=0.9,
    main_title_fontsize=13,
    main_title_fontweight="bold",
    sub_title_fontsize=10,
    legend_fontsize=9,
    x_ticks_rotation=45,
    x_label_fontsize=10,
    y_label_fontsize=10,
    include_legend=True,
    drop_legend_when_n_features=15,
):
    # `n_components` must be less than or equal to the number of components in the PCA
    if n_components > pca.n_components_:
        n_components = pca.n_components_

    # cumulative loading threshold is the average loading threshold times the number of components
    cumulative_loading_threshold = average_loading_threshold * n_components

    # Decide if there will be a legend
    if max_features is None:
        max_features = feature_names.shape[0]

    # Override the legend if there are too many features
    if max_features < drop_legend_when_n_features:
        pass
    else:
        include_legend = False

        # Get the loadings
    loadings = pca.components_.T
    loadings = np.array([x[:n_components] for x in loadings])

    # Create a dataframe with the loadings
    df = pd.DataFrame(
        loadings,
        columns=[f"PC-{'0' if i < 9 else ''}{i+1}" for i in range(n_components)],
        index=feature_names,
    )

    # Absolute value of the loadings to tighten the plot
    df = df.abs()

    # Get the number of features
    total_features = df.shape[0]
    filtered_features = False

    # Sort the loadings by the sum of the first `n_components` columns
    df["sort_col"] = df.cumsum(axis=1).iloc[:, n_components - 1]

    # Test that sort_col is a date or datetime column
    df = df.sort_values(by="sort_col", ascending=False)
    df = df.loc[df.sort_col > cumulative_loading_threshold, :]
    df = df.drop(columns=["sort_col"])
    hidden_features = total_features - df.shape[0]
    if hidden_features > 0:
        filtered_features = True
        filter_type = "loading_threshold"

    # Only show the top `max_features` features, or all features if there are less than `max_features`
    if max_features is None:
        max_features = total_features
    else:
        if df.shape[0] > max_features:
            df = df.iloc[:max_features, :]
            filtered_features = True
            filter_type = "max_features"
            hidden_features = total_features - max_features

    # Get the explained variance for the first `n_components` components (for the title)
    explained_variance = pca.explained_variance_ratio_[:n_components].sum()

    # Test to show if any features were hidden
    hidden_features_text = (
        f" - {hidden_features} features are hidden" if hidden_features > 0 else ""
    )

    # Labels and other text elements will be the same regardless of backend
    x_label_no_filter = "Features" + (
        f" (Showing all with an ave loading for each of the {n_components} components > {average_loading_threshold:.2f}{hidden_features_text})"
        if filtered_features
        else ""
    )
    x_label_filter = "Features" + (
        (
            f" (Only showing the top {max_features} features{hidden_features_text})"
            if filter_type == "max_features"
            else f" (Showing all with an ave loading for each of the {n_components} components > {average_loading_threshold:.2f}{hidden_features_text})"
        )
        if filtered_features
        else ""
    )

    y_label_text = "Cumulative Absolute Loading"

    main_title_text = "Cumulative Influence On Explained Variance"
    sub_title_text = f"The cumulative absolute value of the loadings for each feature for the first {n_components} principal components\nThis plot indicates the features' relative contributions to the {explained_variance:.1%} of variance explained by the {n_components} components"

    # If the backend is plotly, raise a not implemented error
    if backend == "plotly":
        raise NotImplementedError("Plotly support is not yet implemented")
    elif backend != "matplotlib":
        raise ValueError(f"Backend {backend} is not supported")
    else:
        # Create a new figure and axes if needed
        if ax is None:
            _, ax = plt.subplots(figsize=figsize)

        # Plot the loadings for the first `n_components` components as a stacked bar plot
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

        ax.set_xlabel(
            x_label_filter if filtered_features else x_label_no_filter,
            fontsize=x_label_fontsize,
        )
        ax.set_ylabel(y_label_text, fontsize=y_label_fontsize)

        plt.suptitle(
            main_title_text,
            fontsize=main_title_fontsize,
            fontweight=main_title_fontweight,
        )

        ax.set_title(
            sub_title_text,
            fontsize=sub_title_fontsize,
        )
        plt.xticks(rotation=x_ticks_rotation, ha="right")

        # Add a legend if needed
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
