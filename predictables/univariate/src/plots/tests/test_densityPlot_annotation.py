import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  # type: ignore
import pytest

from predictables.univariate.src.plots._density_plot import _annotate_mean_median
from predictables.util import to_pl_s


@pytest.fixture
def feature_pd():
    return pd.Series([1, 2, 3, 4, 5, 6])


@pytest.fixture
def target_pd():
    return pd.Series([0, 0, 1, 1, 0, 1])


@pytest.fixture
def feature_np():
    return np.array([1, 2, 3, 4, 5, 6])


@pytest.fixture
def target_np():
    return np.array([0, 0, 1, 1, 0, 1])


@pytest.fixture
def feature_pl():
    return to_pl_s([1, 2, 3, 4, 5, 6])


@pytest.fixture
def target_pl():
    return to_pl_s([0, 0, 1, 1, 0, 1])


@pytest.fixture
def pd_(feature_pd, target_pd):
    return feature_pd, target_pd


@pytest.fixture
def np_(feature_np, target_np):
    return feature_np, target_np


@pytest.fixture
def pl_(feature_pl, target_pl):
    return feature_pl, target_pl


@pytest.mark.parametrize("lib_f", ["pd", "np", "pl"])
@pytest.mark.parametrize("lib_t", ["pd", "np", "pl"])
def test_basic_functionality_ax_incl(
    feature_pd,
    feature_np,
    feature_pl,
    target_pd,
    target_np,
    target_pl,
    lib_t,
    lib_f,
):
    if lib_f == "pd":
        feature = feature_pd
    elif lib_f == "np":
        feature = feature_np
    elif lib_f == "pl":
        feature = feature_pl

    if lib_t == "pd":
        target = target_pd
    elif lib_t == "np":
        target = target_np
    elif lib_t == "pl":
        target = target_pl

    # Execute the function
    _, ax = plt.subplots()
    ax = _annotate_mean_median(
        feature,
        target,
        ax,
    )

    # Check if the function returns a matplotlib.axes.Axes object
    assert isinstance(ax, plt.Axes), (
        "The function should return a matplotlib.axes.Axes object, "
        f"but returned {type(ax)}"
    )

    # Close the plot
    plt.close("all")


@pytest.mark.parametrize("lib_f", ["pd", "np", "pl"])
@pytest.mark.parametrize("lib_t", ["pd", "np", "pl"])
def test_basic_functionality_ax_not_incl(
    feature_pd,
    feature_np,
    feature_pl,
    target_pd,
    target_np,
    target_pl,
    lib_t,
    lib_f,
):
    if lib_f == "pd":
        feature = feature_pd
    elif lib_f == "np":
        feature = feature_np
    elif lib_f == "pl":
        feature = feature_pl

    if lib_t == "pd":
        target = target_pd
    elif lib_t == "np":
        target = target_np
    elif lib_t == "pl":
        target = target_pl

    # Execute the function
    ax = _annotate_mean_median(
        feature,
        target,
    )

    # Check if the function returns a matplotlib.axes.Axes object
    assert isinstance(ax, plt.Axes), (
        "The function should return a matplotlib.axes.Axes object, "
        f"but returned {type(ax)}"
    )

    # Close the plot
    plt.close("all")


def test_different_lengths():
    fig, ax = plt.subplots()
    with pytest.raises(ValueError) as err:
        _annotate_mean_median(
            pd.Series([1, 2, 3, 4, 5, 6]),
            pd.Series([0, 0, 1, 1, 0]),
            ax,
        )
        plt.close("all")
    assert "The feature and target variables must be the same length" in str(
        err.value
    ), (
        "The function should raise a ValueError when the feature and target series "
        f"have different lengths, but raised {err.value}"
    )


@pytest.mark.parametrize(
    "feature, target",
    [
        (pd.Series([]), pd.Series([])),  # Empty dataset
        (pd.Series([1, 2, np.nan, 4]), pd.Series([0, 1, 0, 1])),  # Dataset with NaN
    ],
)
def test_edge_cases(feature, target):
    fig, ax = plt.subplots()
    with pytest.raises(ValueError) as err:
        _annotate_mean_median(feature, target, ax)
        plt.close("all")
    assert (
        "The feature and target series should not contain NaN or missing values"
        in str(err.value)
    ), (
        "The function should raise a ValueError when the feature or target series "
        f"contain NaN values, but raised {err.value}"
    )


# def test_invalid_inputs():
#     fig, ax = plt.subplots()
#     with pytest.raises(TypeError):
#         _annotate_mean_median("not an ax", "not a series", "also not a series")
