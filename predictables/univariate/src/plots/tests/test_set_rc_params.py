import pytest

from predictables.univariate.src.plots.util._get_rc_params import get_rc_params
from predictables.univariate.src.plots.util._set_rc_params import set_rc_params


@pytest.mark.parametrize(
    "rcParams, expected",
    [
        ({"font.size": 12}, {"font.size": 12, **get_rc_params()}),
        ({"font.size": 45}, {"font.size": 45, **get_rc_params()}),
        ({"legend.fontsize": 12}, {"legend.fontsize": 12, **get_rc_params()}),
        ({"legend.fontsize": 45}, {"legend.fontsize": 45, **get_rc_params()}),
        ({"figure.figsize": (8, 6)}, {"figure.figsize": (8, 6), **get_rc_params()}),
        ({"figure.figsize": (12, 10)}, {"figure.figsize": (12, 10), **get_rc_params()}),
        ({"figure.dpi": 100}, {"figure.dpi": 100, **get_rc_params()}),
        ({"figure.dpi": 200}, {"figure.dpi": 200, **get_rc_params()}),
        (
            {
                "font.size": 12,
                "legend.fontsize": 12,
                "figure.figsize": (8, 6),
                "figure.dpi": 100,
            },
            {
                "font.size": 12,
                "legend.fontsize": 12,
                "figure.figsize": (8, 6),
                "figure.dpi": 100,
                **get_rc_params(),
            },
        ),
        (
            {
                "font.size": 45,
                "legend.fontsize": 45,
                "figure.figsize": (12, 10),
                "figure.dpi": 200,
            },
            {
                "font.size": 45,
                "legend.fontsize": 45,
                "figure.figsize": (12, 10),
                "figure.dpi": 200,
                **get_rc_params(),
            },
        ),
    ],
)
def test_set_rc_params(rcParams, expected):
    updated_rcParams = {**set_rc_params(rcParams)}
    assert (
        updated_rcParams == expected
    ), f"Expected {expected}, but got {updated_rcParams}"
