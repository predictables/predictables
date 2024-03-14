import numpy as np
import pytest

from predictables.univariate.src.plots.util._binary_color import binary_color


@pytest.mark.parametrize(
    "input_value, expected_color",
    [
        # Blues: either 0 or -1 (depending on how the variable is coded)
        (0, "blue"),
        (0.0, "blue"),
        ("0", "blue"),
        (-1, "blue"),
        (-1.0, "blue"),
        ("-1", "blue"),
        (False, "blue"),
        ("false", "blue"),
        ("False", "blue"),
        ("FALSE", "blue"),
        ("f", "blue"),
        ("F", "blue"),
        ("no", "blue"),
        ("n", "blue"),
        ("N", "blue"),
        ("NO", "blue"),
        ("nO", "blue"),
        (1, "orange"),
        (1.0, "orange"),
        ("1", "orange"),
        ("+1", "orange"),
        (True, "orange"),
        ("true", "orange"),
        ("True", "orange"),
        ("TRUE", "orange"),
        ("t", "orange"),
        ("T", "orange"),
        ("yes", "orange"),
        ("y", "orange"),
        ("Y", "orange"),
        ("YES", "orange"),
        ("yEs", "orange"),
    ],
)
def test_binary_color_valid(input_value, expected_color):
    """
    Test the binary_color function with valid inputs to ensure
    it returns the correct color.
    """
    assert binary_color(input_value) == expected_color, (
        f"Expected {expected_color} for {input_value}, but got "
        f"{binary_color(input_value)}"
    )


@pytest.mark.parametrize("invalid_value", [2, 100, 2.0, 100.0, 2.5, "a", None])
def test_binary_color_invalid(invalid_value):
    """
    Test the binary_color function with invalid inputs to ensure
    it raises ValueError.
    """
    with pytest.raises(ValueError) as e:
        binary_color(invalid_value)
        if isinstance(invalid_value, str):
            assert f"Invalid value {invalid_value} for binary variable." in str(e.value)
        else:
            assert f"Invalid value {int(invalid_value)} for binary variable." in str(
                e.value
            )


@pytest.mark.parametrize("x", [0, 1, -1])
def test_numpy_ints(x):
    x_ = np.int64(x)
    _x = np.int32(x)
    assert binary_color(x_) == binary_color(x), f"Failed for np.int64({x})"
    assert binary_color(_x) == binary_color(x), f"Failed for np.int32({x})"


@pytest.mark.parametrize("x", [0.0, 1.0, -1.0])
def test_numpy_floats(x):
    x_ = np.float64(x)
    _x = np.float32(x)
    assert binary_color(x_) == binary_color(x), f"Failed for np.float64({x})"
    assert binary_color(_x) == binary_color(x), f"Failed for np.float32({x})"
