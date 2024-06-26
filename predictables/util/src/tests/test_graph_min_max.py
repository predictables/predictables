import numpy as np
import pandas as pd
import polars as pl
import pytest

from predictables.util.src._graph_min_max import graph_min_max


@pytest.fixture
def x_pd():
    rg = np.random.Generator(np.random.PCG64(42))
    return pd.Series(np.sort(rg.normal(size=1000)))


@pytest.fixture
def x_np(x_pd: pd.Series):
    return x_pd.to_numpy()


@pytest.fixture
def x_pl(x_pd: pd.Series):
    return pl.from_pandas(x_pd)


@pytest.mark.parametrize("x_type", ["pd", "np", "pl"])
@pytest.mark.parametrize("x_min", [None, 0, -1, -23])
@pytest.mark.parametrize("x_max", [None, 5])
def test_calculate_single_density_sd_basic(
    x_pd: pd.Series,
    x_np: np.ndarray,
    x_pl: pl.Series,
    x_max: float,
    x_min: float,
    x_type: str,
):
    """Test that default min and max are correctly identified from the data."""
    # Set x to the correct type
    if x_type == "pd":
        x = x_pd
    elif x_type == "np":
        x = x_np  # type: ignore[assignment]
    elif x_type == "pl":
        x = x_pl  # type: ignore[assignment]

    # Get the min and max
    min_, max_ = graph_min_max(x, x_min, x_max)

    # If min and max are not provided, should use the min and max of the data
    assert min_ == x_min if x_min is not None else x.min(), (
        f"Since x_min is {'`None`' if x_min is None else 'provided'}, "
        f"min_ ({min_}) should be equal to "
        f"{'`x.min()`' if x_min is None else '`x_min`'} "
        f"({x.min() if x_min is None else x_min})"
    )
    assert max_ == x_max if x_max is not None else x.max(), (
        f"Since x_max is {'`None`' if x_max is None else 'provided'}, "
        f"max_ ({max_}) should be equal to "
        f"{'`x.max()`' if x_max is None else '`x_max`'} "
        f"({x.max() if x_max is None else x_max})"
    )

    assert min_ <= max_, f"min ({min_}) should be less than or equal to max ({max_})"

    assert isinstance(
        min_, (int, float)
    ), f"min_ ({min_}) should be a float or an integer, not {type(min_)}"
    assert isinstance(
        max_, (int, float)
    ), f"max_ ({max_}) should be a float or an integer, not {type(max_)}"

    assert isinstance(graph_min_max(x, x_min, x_max), tuple), (
        "The function should return a tuple, "
        f"but returned {type(graph_min_max(x, x_min, x_max))}"
    )
    assert len(graph_min_max(x, x_min, x_max)) == 2, (
        "The function should return a tuple of length 2, "
        f"but returned {len(graph_min_max(x, x_min, x_max))}"
    )
    assert isinstance(graph_min_max(x, x_min, x_max)[0], (int, float)), (
        "The first element of the tuple should be a float or an integer, "
        f"but returned {type(graph_min_max(x, x_min, x_max)[0])}"
    )
    assert isinstance(graph_min_max(x, x_min, x_max)[1], (int, float)), (
        "The second element of the tuple should be a float or an integer, "
        f"but returned {type(graph_min_max(x, x_min, x_max)[1])}"
    )


def test_min_gt_max(x_pd: pd.Series):
    x = x_pd
    with pytest.raises(ValueError) as error:
        graph_min_max(x, 1, 0)
    assert "min_ (1) should be less than or equal to max_ (0)" in str(
        error.value
    ), "The function should raise a ValueError if min_ is greater than max_"
