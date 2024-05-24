"""have a pca function that returns a fitted pca object. use it to generate a pca object and then use the pca object to generate a loading plot. the loading plot should be a bar chart with the x-axis as the feature names and the y-axis as the pca loadings. the plot should be interactive and should allow the user to hover over the bars to see the feature names and the pca loadings. the plot should be displayed in the streamlit app."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.decomposition import PCA
from bokeh.plotting import figure
from bokeh.palettes import Viridis
from bokeh.models import ColumnDataSource
from bokeh.transform import factor_cmap


def MAIN_TITLE_TEXT() -> str:
    """Define the main title text for the loading plot."""
    return "Cumulative Influence On Explained Variance"


def SUBTITLE_TEXT(n_components: int, explained_variance: float) -> str:
    """Define the subtitle text for the loading plot."""
    return f"The cumulative absolute value of the loadings for each feature for the first {n_components} principal components\nThis plot indicates the features' relative contributions to the {explained_variance:.1%} of variance explained by the {n_components} components"


def Y_AXIS_LABEL() -> str:
    """Define the y-axis label for the loading plot."""
    return "Cumulative Absolute Loading"


def X_AXIS_LABEL(
    filtered_features: bool,
    n_components: int,
    average_loading_threshold: float,
    hidden_features_text: str,
    max_features: int,
    filter_type: str,
) -> str:
    """Define the x-axis label for the loading plot."""
    if filtered_features:
        return "Features" + (
            f" (Showing all with an ave loading for each of the {n_components} components > {average_loading_threshold:.2f}{hidden_features_text})"
        )
    else:
        return (
            f" (Only showing the top {max_features} features{hidden_features_text})"
            if filter_type == "max_features"
            else f" (Showing all with an ave loading for each of the {n_components} components > {average_loading_threshold:.2f}{hidden_features_text})"
        )


def loading_plot(
    pca: PCA,
    n_components: int,
    feature_names: list,
    average_loading_threshold: float = 0.1,
    max_features: int | None = None,
    drop_legend_when_n_features: int = 20,
    include_legend: bool = True,
    bar_width: float = 0.5,
    bar_alpha: float = 0.8,
) -> figure:
    """Generate a loading plot."""
    # `n_components` must be less than or equal to the number of components in the PCA
    n_components = min(n_components, pca.n_components_)

    # cumulative loading threshold is the average loading threshold times the number of components
    cumulative_loading_threshold = average_loading_threshold * n_components

    # Decide if there will be a legend
    if max_features is None:
        max_features = feature_names.shape[0]

    # Override the legend if there are too many features
    if max_features >= drop_legend_when_n_features:
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
    elif df.shape[0] > max_features:
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

    # Initialize the figure
    p = figure(
        title=MAIN_TITLE_TEXT(),
        x_axis_label=X_AXIS_LABEL(
            filtered_features,
            n_components,
            average_loading_threshold,
            hidden_features_text,
            max_features,
            filter_type,
        ),
        y_axis_label=Y_AXIS_LABEL(),
        width=1200,
        height=800,
        x_range=df.index.tolist(),
        tools="hover, box_zoom, pan, reset, save",
        # axis_label_orientation=45,
    )

    if include_legend:
        p.legend.location = "top_right"
        p.legend.click_policy = "hide"
        # p.legend.xlabel_text_rotation = 45
    else:
        p.legend.visible = False

    components = df.columns.tolist()

    source = ColumnDataSource(df.reset_index().rename(columns={"index": "feature"}))

    # Plot the loadings for the first `n_components` components as a stacked bar plot
    v_256_slice = [(256 // n_components) * i for i in range(n_components)]
    p.vbar_stack(
        components,
        x="feature",
        source=source,
        width=bar_width,
        fill_alpha=bar_alpha,
        color=[Viridis[256][i] for i in v_256_slice],
        legend_label=components,
        line_color="black",
        line_width=0.5,

    )

    p.xaxis.major_label_orientation = 0.75
    p.xaxis.major_label_text_font_size = "12pt"
    p.xaxis.major_label_text_align = "right"

    p.hover.tooltips = [
        ("Feature", "@feature"),
        ("PC", "$name"),
        ("Loading", "@$name{0.00}"),
    ]

    return p