import pytest
import numpy as np
from predictables.app.plots.roc.src.plot_generation import generate_roc_auc_plot


def test_generate_roc_auc_plot():
    roc_curves = [
        (
            np.array([0.0, 0.1, 0.4, 0.5, 0.7, 1.0]),
            np.array([0.0, 0.2, 0.6, 0.7, 0.8, 1.0]),
            0.9,
        ),
        (
            np.array([0.0, 0.1, 0.3, 0.4, 0.8, 1.0]),
            np.array([0.0, 0.3, 0.5, 0.7, 0.85, 1.0]),
            0.85,
        ),
    ]
    mean_fpr = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    mean_tpr = np.array([0.0, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.9, 0.95, 1.0])
    std_error = np.array(
        [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05]
    )

    # Use Bokeh's show function to render the plot
    # This test checks if the function runs without errors
    p = generate_roc_auc_plot(roc_curves, mean_fpr, mean_tpr, std_error)

    assert (
        p.title.text == "ROC AUC Plot"
    ), f"Expected 'ROC AUC Plot', got {p.title.text}"
    assert (
        p.xaxis.axis_label == "False Positive Rate"
    ), f"Expected 'False Positive Rate', got {p.xaxis.axis_label}"
    assert (
        p.yaxis.axis_label == "True Positive Rate"
    ), f"Expected 'True Positive Rate', got {p.yaxis.axis_label}"
    assert len(p.renderers) == 2 + 2, f"Expected 4 renderers, got {len(p.renderers)}"
    assert (
        p.legend[0].items[0].label["value"] == "Fold 1 (AUC = 0.90)"
    ), f"Expected 'Fold 1 (AUC = 0.90)', got {p.legend[0].items[0].label['value']}"
    assert (
        p.legend[0].items[1].label["value"] == "Fold 2 (AUC = 0.85)"
    ), f"Expected 'Fold 2 (AUC = 0.85)', got {p.legend[0].items[1].label['value']}"
    assert (
        p.legend[0].items[2].label["value"] == "Mean ROC"
    ), f"Expected 'Mean ROC', got {p.legend[0].items[2].label['value']}"
    assert (
        p.legend[0].items[3].label["value"] == "±1 SE"
    ), f"Expected '±1 SE', got {p.legend[0].items[3].label['value']}"
