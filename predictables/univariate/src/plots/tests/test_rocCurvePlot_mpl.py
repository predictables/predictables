import pytest
import pandas as pd  # type: ignore
import numpy as np
import matplotlib.pyplot as plt
from predictables.univariate.src.plots._roc_curve_plot import roc_curve_plot_mpl


# Fixture for mock data
@pytest.fixture
def mock_data():
    np.random.seed(0)
    size = 100
    y = pd.Series(np.random.randint(0, 2, size))
    yhat_proba = pd.Series(np.random.rand(size))
    fold = pd.Series(np.random.randint(1, 5, size))
    return y, yhat_proba, fold


# Test for successful plot creation
def test_successful_plot_creation(mock_data):
    y, yhat_proba, fold = mock_data
    ax = roc_curve_plot_mpl(y, yhat_proba, fold, False, 0.1, 0.01, 0.05)
    assert isinstance(ax, plt.Axes), (
        "The function should return a matplotlib.axes.Axes object, "
        f"but returned {type(ax)}"
    )
    plt.close("all")


# Test for invalid ax parameter type
def test_invalid_ax_parameter_type(mock_data):
    y, yhat_proba, fold = mock_data
    with pytest.raises(TypeError) as err:
        roc_curve_plot_mpl(y, yhat_proba, fold, False, 0.1, 0.01, 0.05, ax="invalid")
        plt.close("all")
    assert "The ax parameter should be a matplotlib.axes.Axes object" in str(
        err.value
    ), (
        "The function should raise a TypeError when the ax parameter is not a "
        f"matplotlib.axes.Axes object, but raised {err.value}"
    )


# Test with and without ax parameter
def test_with_without_ax_parameter(mock_data):
    y, yhat_proba, fold = mock_data
    # Without ax
    ax1 = roc_curve_plot_mpl(y, yhat_proba, fold, False, 0.1, 0.01, 0.05)
    assert isinstance(ax1, plt.Axes), (
        "The function should work without an ax parameter, " f"but returned {type(ax1)}"
    )
    plt.close("all")
    # With ax
    fig, ax2 = plt.subplots()
    ax3 = roc_curve_plot_mpl(y, yhat_proba, fold, False, 0.1, 0.01, 0.05, ax=ax2)
    assert ax3 is ax2, (
        "The function should use the provided ax parameter, "
        f"but returned a different ax object ({type(ax3)})"
    )
    plt.close("all")


@pytest.mark.parametrize("y", [pd.Series([1, 0, 1, 0, 1, 1, 0, 0])])
@pytest.mark.parametrize(
    "yhat_proba", [pd.Series([0.9, 0.1, 0.8, 0.2, 0.9, 0.9, 0.2, 0.1])]
)
@pytest.mark.parametrize("fold", [pd.Series([1, 2, 1, 2, 1, 2, 1, 2])])
@pytest.mark.parametrize("time_series_validation", [True, False])
@pytest.mark.parametrize("coef", [0.5, -0.5])
@pytest.mark.parametrize("se", [0.01])
@pytest.mark.parametrize("pvalue", [0.001, 0.5])
@pytest.mark.parametrize("figsize", [(7, 7)])
@pytest.mark.parametrize("n_bins", [100, 200, 300])
@pytest.mark.parametrize("cv_alpha", [0.1, 1.0, None])
@pytest.mark.parametrize("ax", [None, plt.subplots()[1]])
def test_parameterized_inputs(
    y,
    yhat_proba,
    fold,
    time_series_validation,
    coef,
    se,
    pvalue,
    figsize,
    n_bins,
    cv_alpha,
    ax,
):
    def plot_generator():
        yield roc_curve_plot_mpl(
            y,
            yhat_proba,
            fold,
            time_series_validation,
            coef,
            se,
            pvalue,
            figsize,
            n_bins,
            cv_alpha,
            ax,
        )
        plt.close("all")

    ax = next(plot_generator())
    assert isinstance(ax, plt.Axes), (
        f"The function should handle different inputs correctly, but returned {type(ax)} "
        f"when y={y}, yhat_proba={yhat_proba}, fold={fold}, time_series_validation={time_series_validation}, coef={coef}, se={se}, pvalue={pvalue}, figsize={figsize}, n_bins={n_bins}, cv_alpha={cv_alpha}, ax={ax}"
    )
    plt.close("all")
