import numpy as np
import pandas as pd
import pytest

from predictables.impute.src._impute_with_mode import _is_mode_ambiguous
from predictables.util import to_pl_s


@pytest.fixture
def ndarray_2d_array():
    # Create a numpy array with different values
    return np.array([[1, 2, 2], [3, 3, 3]])


def ambiguous():
    return pd.Series([1, 2, 2, 3, 3, 4, 4])


def unambiguous():
    return pd.Series([1, 2, 2, 2, 3, 4, 4])  # mode of 2


def ambiguous_str():
    return pd.Series(["a", "b", "b", "c", "c"])


def unambiguous_str():
    return pd.Series(["a", "b", "b", "b", "c"])  # mode of 'b'


def ambiguous_cat():
    return (
        pd.Series(["a", "b", "b", "c", "c"]).astype("category").reset_index(drop=True)
    )


def unambiguous_cat():
    return (
        pd.Series(["a", "b", "b", "b", "c"]).astype("category").reset_index(drop=True)
    )  # mode of 'b'


mark_vals_is_mode_ambiguous = [
    ("pd", ambiguous(), True),
    ("pl", to_pl_s(ambiguous()), True),
    ("np", ambiguous().to_numpy(), True),
    ("pd", unambiguous(), False),
    ("pl", to_pl_s(unambiguous()), False),
    # ^ 5
    ("np", unambiguous().to_numpy(), False),
    ("pd", ambiguous_str(), True),
    ("pl", to_pl_s(ambiguous_str()), True),
    ("np", ambiguous_str().to_numpy(), True),
    ("pd", unambiguous_str(), False),
    # ^ 10
    ("pl", to_pl_s(unambiguous_str()), False),
    ("np", unambiguous_str().to_numpy(), False),
    ("pd", ambiguous_cat(), True),
    ("pl", to_pl_s(ambiguous_cat()), True),
    ("np", ambiguous_cat().to_numpy(), True),
    # ^ 15
    ("pd", unambiguous_cat(), False),
    ("pl", to_pl_s(unambiguous_cat()), False),
    ("np", unambiguous_cat().to_numpy(), False),
]

first_mode_data = [
    ("pd", ambiguous(), 2),
    ("pl", to_pl_s(ambiguous()), 2),
    ("np", ambiguous().to_numpy(), 2),
    ("pd", ambiguous_str(), "b"),
    ("pl", to_pl_s(ambiguous_str()), "b"),  # ERRORS
    # ^ 5
    ("np", ambiguous_str().to_numpy(), "b"),
    ("pd", ambiguous_cat(), "b"),
    ("pl", to_pl_s(ambiguous_cat()), "b"),  # ERRORS
    ("np", ambiguous_cat().to_numpy(), "b"),
]


@pytest.mark.parametrize("lib,s,expected", mark_vals_is_mode_ambiguous)
def test_is_mode_ambiguous_normal_input(lib, s, expected):
    """Testing that a pandas series that unambiguously has a mode of 3 is correctly detected"""
    assert (
        _is_mode_ambiguous(s) is expected
    ), f"Expected that the mode is {'' if expected else 'not '}ambiguous, but it is{' not' if expected else ''}.\nShape: {s.mode().shape}Mode:\n{s.mode()}"


def test_is_mode_ambiguous_with_numpy_2d_array_gives_error(ndarray_2d_array):
    s = ndarray_2d_array
    """Testing that a numpy array that is 2d gives an error, even though there is unambiguously a mode of 3 if the two dimensions were to be squished into one"""
    with pytest.raises(ValueError):
        (
            _is_mode_ambiguous(s),
            f"Expected that the running `_is_mode_ambiguous` on a 2d numpy array would raise a ValueError, but it did not:\n\n_is_mode_ambiguous(ndarray_2d_array):\n{_is_mode_ambiguous(s)}",
        )
