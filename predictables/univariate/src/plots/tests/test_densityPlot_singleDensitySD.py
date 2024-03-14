import numpy as np
import pandas as pd  # type: ignore
import polars as pl
import pytest

from predictables.univariate.src.plots._density_plot import _calculate_single_density_sd


@pytest.fixture
def x_pd():
    np.random.seed(42)
    return pd.Series(np.random.randn(100))


@pytest.fixture
def x_np(x_pd):
    return x_pd.values


@pytest.fixture
def x_pl(x_pd):
    return pl.from_pandas(x_pd)


@pytest.fixture
def cv_pd():
    return pd.Series(np.random.choice([0, 1, 2, 3], size=100))


@pytest.fixture
def cv_np(cv_pd):
    return cv_pd.values


@pytest.fixture
def cv_pl(cv_pd):
    return pl.from_pandas(cv_pd)


@pytest.mark.parametrize("time_series_validation", [True, False])
@pytest.mark.parametrize("grid_bins", [100, 200, 50])
@pytest.mark.parametrize("x_type", ["pd", "np", "pl"])
@pytest.mark.parametrize("cv_type", ["pd", "np", "pl"])
def test_calculate_single_density_sd_basic(
    x_pd,
    x_np,
    x_pl,
    cv_pd,
    cv_np,
    cv_pl,
    cv_type,
    x_type,
    grid_bins,
    time_series_validation,
):
    if x_type == "pd":
        x = x_pd
    elif x_type == "np":
        x = x_np
    elif x_type == "pl":
        x = x_pl

    if cv_type == "pd":
        cv_fold = cv_pd
    elif cv_type == "np":
        cv_fold = cv_np
    elif cv_type == "pl":
        cv_fold = cv_pl

    # Expected behavior: function runs without errors and returns a Series
    sd_smooth, sd = _calculate_single_density_sd(
        x, cv_fold, grid_bins, time_series_validation
    )

    assert isinstance(
        sd_smooth, pd.Series
    ), f"Expected sd_smooth to be a pandas Series, but got {type(sd_smooth)}"
    assert (
        len(sd_smooth) == grid_bins
    ), f"Expected length of sd_smooth to be {grid_bins}, but got {len(sd_smooth)}"


@pytest.mark.parametrize("grid_bins", [10, 50, 100])
def test_calculate_single_density_sd_edge_cases(grid_bins):
    x = pd.Series(np.random.randn(10))
    cv_fold = pd.Series(np.random.choice([0, 1], size=10))

    sd_smooth, sd = _calculate_single_density_sd(x, cv_fold, grid_bins)

    assert (
        len(sd_smooth) == grid_bins
    ), f"Expected length of sd_smooth to be {grid_bins} for edge case"


@pytest.mark.parametrize(
    "x,cv_fold",
    [
        (pd.Series([]), pd.Series([])),
        (pd.Series([1, 2, 3, 4]), pd.Series([])),
        (pd.Series([]), pd.Series([1, 2, 1, 2])),
    ],
)
def test_calculate_single_density_sd_empty_input(x, cv_fold):
    with pytest.raises(ValueError):
        _calculate_single_density_sd(x, cv_fold)


def test_calculate_single_density_sd_uniform_input():
    x = pd.Series([1] * 100)
    cv_fold = pd.Series([0] * 50 + [1] * 50)

    with pytest.raises(np.linalg.LinAlgError) as err:
        sd_smooth, sd = _calculate_single_density_sd(x, cv_fold)
    assert (
        "lower-dimensional subspace" in str(err.value).lower()
    ), f"Expected LinAlgError, but got {err}"
