import numpy as np
import pandas as pd  # type: ignore
import polars as pl
import pytest
from scipy.stats import gaussian_kde  # type: ignore
from predictables.univariate.src.plots._density_plot import calculate_density_sd
from predictables.util import filter_by_cv_fold


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
def by_pd():
    return pd.Series(np.random.choice(["A", "B"], size=100))


@pytest.fixture
def by_np(by_pd):
    return by_pd.values


@pytest.fixture
def by_pl(by_pd):
    return pl.from_pandas(by_pd)


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
@pytest.mark.parametrize("grid_bins", [50])
@pytest.mark.parametrize("x_type", ["pd", "np", "pl"])
@pytest.mark.parametrize("by_type", ["pd", "np", "pl"])
@pytest.mark.parametrize("cv_type", ["pd", "np", "pl"])
@pytest.mark.parametrize("x_min", [None, -1])
@pytest.mark.parametrize("x_max", [None, 1])
def test_calculate_single_density_sd_basic(
    x_pd,
    x_np,
    x_pl,
    by_pd,
    by_np,
    by_pl,
    cv_pd,
    cv_np,
    cv_pl,
    x_max,
    x_min,
    cv_type,
    by_type,
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

    if by_type == "pd":
        by = by_pd
    elif by_type == "np":
        by = by_np
    elif by_type == "pl":
        by = by_pl

    if cv_type == "pd":
        cv_fold = cv_pd
    elif cv_type == "np":
        cv_fold = cv_np
    elif cv_type == "pl":
        cv_fold = cv_pl

    sd_smooth, sd = calculate_density_sd(
        x,
        by,
        cv_fold,
        x_min,
        x_max,
        grid_bins,
        time_series_validation,
    )

    # Basic checks
    assert isinstance(sd_smooth, pd.Series), (
        "Expected sd_smooth to be a pandas Series, but got " f"{type(sd_smooth)}"
    )
    assert isinstance(sd, pd.Series), (
        "Expected sd to be a pandas Series, but got " f"{type(sd)}"
    )
    assert (
        len(sd_smooth) == grid_bins
    ), f"Expected length of sd_smooth to be {grid_bins}, but got {len(sd_smooth)}"
    assert (
        sd >= 0
    ).all(), f"Expected all SD values to be non-negative, but there were {sum(sd < 0)} negative values"

    # Check calculated SD is within 2 SD at least 90% of the time (to be conservative)
    sd_2 = 2 * sd_smooth
    sd_2 = sd_2.reindex(sd.index, method="nearest")
    assert (sd <= sd_2).mean() > 0.9, (
        "Expected at least 90% of the calculated SD to be within 2 SD, but got "
        f"{(sd <= sd_2).mean()}"
    )

    # Check that the smoothed SD is also within 2 SD at least 90% of the time
    sd_smooth_2 = 2 * sd_smooth
    assert (sd_smooth <= sd_smooth_2).mean() > 0.9, (
        "Expected at least 90% of the smoothed SD to be within 2 SD, but got "
        f"{(sd_smooth <= sd_smooth_2).mean()}"
    )


# def test_calculate_density_sd_cv_fold_none():
#     x = pd.Series(np.random.randn(10))
#     by = pd.Series(np.random.choice(["A", "B"], size=10))

#     with pytest.raises(ValueError):
#         calculate_density_sd
