import numpy as np
import pytest

from predictables.util.src.harmonic_mean import harmonic_mean


def test_harmonic_mean_single_value():
    # Harmonic mean of a single value should be the value itself
    assert harmonic_mean(5) == 5, f"Expected 5, got {harmonic_mean(5)}"


def test_harmonic_mean_two_values():
    result = 2 / ((1 / 2) + (1 / 6))
    assert np.isclose(
        harmonic_mean(2, 6), result
    ), f"Expected {result}, got {harmonic_mean(2, 6)}"


def test_harmonic_mean_multiple_values():
    result = 3 / ((1 / 1) + (1 / 2) + (1 / 4))
    assert np.isclose(
        harmonic_mean(1, 2, 4), result
    ), f"Expected {result}, got {harmonic_mean(1, 2, 4)}"


def test_harmonic_mean_multiple_values_with_zero():
    result = 4 / ((1 / 1) + (1 / 2) + (1 / 4))
    assert np.isclose(
        harmonic_mean(1, 2, 4, 0), result
    ), f"Expected {result}, got {harmonic_mean(1, 2, 4, 0)}"


def test_multiple_values_btwn_zero_and_one():
    values = [0.5, 0.25, 0.125, 0.0625, 0.75, 0.375, 0.1875, 0.09375]
    result = len(values) / (
        (1 / 0.5)
        + (1 / 0.25)
        + (1 / 0.125)
        + (1 / 0.0625)
        + (1 / 0.75)
        + (1 / 0.375)
        + (1 / 0.1875)
        + (1 / 0.09375)
    )
    assert np.isclose(
        harmonic_mean(*values), result
    ), f"Expected {result}, got {harmonic_mean(*values)}"


def test_harmonic_mean_zero():
    result = 0
    assert np.isclose(
        harmonic_mean(0), result
    ), f"Expected {result}, got {harmonic_mean(0)}"


def test_harmonic_mean_negative():
    result = 2 / ((1 / -1) + (1 / -2))
    assert np.isclose(
        harmonic_mean(-1, -2), result
    ), f"Expected {result}, got {harmonic_mean(-1, -2)}"
