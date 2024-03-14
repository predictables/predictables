import numpy as np
import pandas as pd
import polars as pl
import pytest

from predictables.univariate.src.plots._cdf_plot import calculate_cdf


@pytest.mark.parametrize(
    "x, expected",
    [
        (np.array([1, 2, 3, 4, 5]), np.array([0.2, 0.4, 0.6, 0.8, 1.0])),
        (pd.Series([1, 2, 3, 4, 5]), np.array([0.2, 0.4, 0.6, 0.8, 1.0])),
        (pl.Series([1, 1, 1, 1, 1]), np.array([0.2, 0.4, 0.6, 0.8, 1.0])),
        (
            np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
            np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]),
        ),
        (np.array([1, 1]), np.array([0.5, 1])),
        (np.array([1, 1, 1]), np.array([0.33333333, 0.66666667, 1.0])),
        (np.array([1, 1, 1, 1]), np.array([0.25, 0.5, 0.75, 1.0])),
        (np.array([1, 1, 1, 1, 1]), np.array([0.2, 0.4, 0.6, 0.8, 1.0])),
        (
            np.array([1, 1, 1, 1, 1, 1]),
            np.array([0.16666667, 0.33333333, 0.5, 0.66666667, 0.83333333, 1.0]),
        ),
        (
            np.array([1, 1, 1, 1, 1, 1, 1]),
            np.array(
                [
                    0.14285714,
                    0.28571429,
                    0.42857143,
                    0.57142857,
                    0.71428571,
                    0.85714286,
                    1.0,
                ]
            ),
        ),
    ],
)
def test_calculate_cdf(x, expected):
    cdf = calculate_cdf(x)
    assert np.allclose(
        cdf, expected
    ), f"The simple array {x} should have a cdf given by {expected} but was {cdf}"


@pytest.mark.parametrize(
    "x, msg",
    [
        (np.array([]), "The array must not be empty."),
        (
            np.array([1.1, 2.1, 3.1, 4.1, 5.1, np.nan]),
            "The array must not contain NaNs.",
        ),
        (
            np.array([1.2, 2.2, 3.2, 4.2, 5.2, np.inf]),
            "The array must not contain infs.",
        ),
        (
            np.array([1.2, 2.2, 3.3, 4.3, 5.3, -np.inf]),
            "The array must not contain infs.",
        ),
        (
            np.array(["a", "b", "c", "d"]),
            "The array must not contain non-numeric values.",
        ),
        (
            np.array(["1", "2", "3", "4", "5", ""]),
            "The array must not contain non-numeric values.",
        ),
        (
            np.array(["1", "2", "3", "4", "5", " "]),
            "The array must not contain non-numeric values.",
        ),
        (
            np.array(["1", "2", "3", "4", "5", "6.0"]),
            "The array must not contain non-numeric values.",
        ),
    ],
)
def test_calculate_cdf_invalid_input(x, msg):
    with pytest.raises(ValueError) as e:
        calculate_cdf(x)
    assert str(e.value) == msg, f"Expected error message of {msg} but got {e.value}"
