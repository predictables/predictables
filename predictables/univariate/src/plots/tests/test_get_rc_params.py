import pytest

from predictables.univariate.src.plots.util._get_rc_params import get_rc_params


@pytest.fixture
def rc_params():
    return get_rc_params()


@pytest.mark.parametrize(
    "param,expected",
    [
        ("font.size", 12),
        ("axes.titlesize", 16),
        ("axes.labelsize", 14),
        ("xtick.labelsize", 14),
        ("ytick.labelsize", 14),
        ("figure.titlesize", 16),
        ("figure.figsize", (7, 7)),
        ("figure.dpi", 150),
    ],
)
def test_get_rc_params(rc_params, param, expected):
    assert (
        rc_params[param] == expected
    ), f"Expected {param} to be {expected} but got {rc_params[param]}."
