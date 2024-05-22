"""Generate the ROC-AUC plot."""

from __future__ import annotations
from bokeh.plotting import figure
from bokeh.models import HoverTool
import numpy as np


def generate_roc_auc_plot(
    roc_curves: list[tuple[np.ndarray, np.ndarray, float]],
    mean_fpr: np.ndarray,
    mean_tpr: np.ndarray,
    std_error: np.ndarray,
) -> figure:
    """Generate the ROC AUC plot using Bokeh.

    Parameters
    ----------
    roc_curves : List of tuples
        ROC curves (FPR, TPR) for each fold.
    mean_fpr : np.ndarray
        Mean FPR values.
    mean_tpr : np.ndarray
        Mean TPR values.
    std_error : np.ndarray
        Standard error of TPR values.

    Returns
    -------
    figure
        Bokeh figure object.
    """
    p = figure(
        title="ROC AUC Plot",
        x_axis_label="False Positive Rate",
        y_axis_label="True Positive Rate",
        plot_width=750,
        plot_height=450,
    )

    # Plot ROC curves for each fold
    lines = []
    for i, (fpr, tpr, roc_auc) in enumerate(roc_curves):
        lines.append(
            p.line(
                fpr,
                tpr,
                legend_label=f"Fold {i+1} (AUC = {roc_auc:.2f})",
                line_width=3,
                alpha=0.7,
            )
        )

    # Plot mean ROC curve
    p.line(mean_fpr, mean_tpr, legend_label="Mean ROC", line_width=2, color="black")

    # Plot standard error as shaded area
    p.patch(
        np.concatenate([mean_fpr, mean_fpr[::-1]]),
        np.concatenate([mean_tpr - std_error, (mean_tpr + std_error)[::-1]]),
        color="gray",
        fill_alpha=0.2,
        legend_label="±1 SE",
    )

    # Add 45-degree line
    diag_line = p.line(
        [0, 1],
        [0, 1],
        line_dash="dashed",
        line_width=1,
        color="red",
        legend_label="Line of Equality",
    )
    p.add_layout(diag_line)

    # Add hover tool
    hover = HoverTool()
    hover.tooltips = [("FPR", "$x"), ("TPR", "$y"), ("Fold", "@legend_label")]
    p.add_tools(hover)

    p.legend.location = "bottom_right"

    return p