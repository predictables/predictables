from __future__ import annotations

import streamlit as st

import numpy as np
import pandas as pd
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, LabelSet, Label

from predictables.app.src.pca import pca as _pca


def interpolate(x1: float, x2: float, y1: float, y2: float, y: float) -> float:
    """Calculate the linear interpolation of y given two points (x1, y1) and (x2, y2)."""
    return x1 + (x2 - x1) * (y - y1) / (y2 - y1)


def scree_plot(
    X: pd.DataFrame,
    variance_levels: list[float] | None = None,
    y_pos_adjustment: float = 0.1,
    **kwargs,
) -> figure:
    """Create a scree plot for the given data."""
    if variance_levels is None:
        variance_levels = [0.8, 0.9, 0.95, 0.99]

    pca = _pca(X)
    variance_ratios = pca.explained_variance_ratio_
    cumulative_variance_explained = np.cumsum(variance_ratios)

    n_features = X.shape[1]

    components_required_for_each_level = []

    p = figure(
        x_axis_label="Principal Component",
        y_axis_label="Cumulative Variance Explained",
        width=1000,
        height=600,
        **kwargs,
    )

    p.line(
        x=range(1, n_features + 1),
        y=cumulative_variance_explained,
        line_width=2,
        line_color="blue",
    )

    p.line(
        x=range(2),
        y=[0, cumulative_variance_explained[0]],
        line_width=2,
        line_color="blue",
    )

    p.circle(
        x=range(1, n_features + 1),
        y=cumulative_variance_explained,
        size=8,
        fill_color="blue",
        line_color="black",
        line_width=1,
        fill_alpha=0.5,
    )

    p.circle(
        x=range(1),
        y=[0],
        size=8,
        fill_color="blue",
        line_color="black",
        line_width=1,
        fill_alpha=0.5,
    )

    labx, laby, labels = [], [], []
    for level in variance_levels:
        n_components_required = (
            np.sum(np.less_equal(np.round(cumulative_variance_explained, 4), level)) + 1
        )

        y_position_of_variance_label = level - (y_pos_adjustment) * (
            2 - cumulative_variance_explained[n_components_required - 1]
        )

        components_required_for_each_level.append(n_components_required)

        # Linearly interpolate between points (n_components, level) and
        # (n_components - 1, cumulative_variance[n_components - 2])  # noqa: ERA001
        # in the x direction to find the point where the horizontal line
        # intersects with the cumulative variance curve

        # Horizontal line
        extends_to_x = interpolate(
            n_components_required - 1,
            n_components_required,
            cumulative_variance_explained[n_components_required - 2],
            cumulative_variance_explained[n_components_required - 1],
            level,
        )

        # Vertical line
        extends_to_y = np.interp(
            extends_to_x,
            [n_components_required - 1, n_components_required],
            [
                cumulative_variance_explained[n_components_required - 2],
                cumulative_variance_explained[n_components_required - 1],
            ],
        )

        horizontal_src = ColumnDataSource(
            data={"x": [0, extends_to_x], "y": [extends_to_y, extends_to_y]}
        )

        # Plot the horizontal line
        p.line(
            x="x",
            y="y",
            line_width=2,
            line_color="red",
            line_dash="dashed",
            source=horizontal_src,
        )

        # Plot the vertical line
        p.line(
            x=[extends_to_x, extends_to_x],
            y=[0, extends_to_y],
            line_width=2,
            line_color="green",
            line_dash="dashed",
        )

        # Add a red circle at the intersection point
        p.circle(
            x=[extends_to_x],
            y=[extends_to_y],
            size=10,
            fill_color="red",
            line_color="black",
            line_width=2,
            fill_alpha=0.5,
        )

        labx.append(extends_to_x)
        laby.append(extends_to_y)

        # labels.append(
        #     Label(
        #         x=extends_to_x + 20,
        #         y=extends_to_y + 10,
        #         text=f"{level:.0%}",
        #         # padding=10,
        #         text_font_size="12pt",
        #         text_font_style="bold",
        #         text_color="black",
        #         border_line_color="black",
        #     )
        # )

    label_source = ColumnDataSource(
        data={
            "x": labx,
            "y": laby,
            "label": [
                f"{level:.0%}\n@ {n_components} Components"
                for level, n_components in zip(
                    variance_levels, components_required_for_each_level
                )
            ],
        }
    )

    labels = LabelSet(
        x="x",
        y="y",
        text="label",
        text_font_size="12pt",
        text_font_style="bold",
        text_color="black",
        text_align="right",
        x_offset=-10,
        y_offset=10,
        source=label_source,
        render_mode="css",
        border_line_color="black",
        background_fill_color="white",
        background_fill_alpha=0.9,
    )

    p.add_layout(labels)

    # for label in labels:
    #     p.add_layout(label)

    p.yaxis.bounds = (0, 1)

    scree_interpretation = (
        "The scree plot shows the cumulative variance \n"
        "explained by each principal component. I have \n"
        "added annotations to show the number of components \n"
        "required to explain at least 75%, 90%, 95%, and 99% \n"
        "of the variance. The first component that explains\n"
        "100% of the variance is also annotated. The scree \n"
        "plot is used to help select the number of components \n"
        "to retain. In this case, I would likely retain \n"
        f"{components_required_for_each_level[variance_levels.index(0.9)]:d} "
        "components to retain at least 90% of the variance."
    )

    scree_interpretation_label = Label(
        x=20, y=0, x_units="data", y_units="data", text=scree_interpretation, border_line_color="black", background_fill_color="white", background_fill_alpha=0.9, text_align="right", text_font_size="12pt"
    )

    p.add_layout(scree_interpretation_label)

    return p