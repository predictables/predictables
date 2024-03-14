# noqa: INP001
"""Test the informedness function.

This module contains tests for the informedness function in the
predictables.util.stats module.
"""

import numpy as np

from predictables.util.stats.src._informedness import informedness


def test_informedness_perfect_classifier() -> None:
    """Test the informedness function with a perfect classifier."""
    y = np.array([0, 0, 1, 1])
    yhat = np.array([0, 0, 1, 1])
    assert informedness(y, yhat) == 1, f"Expected 1, got {informedness(y, yhat)}"


def test_informedness_completely_wrong_classifier() -> None:
    """Test the informedness function with a completely wrong classifier."""
    y = np.array([0, 0, 1, 1])
    yhat = np.array([1, 1, 0, 0])
    assert informedness(y, yhat) == -1, f"Expected -1, got {informedness(y, yhat)}"


def test_informedness_general_case() -> None:
    """Test the informedness function with a general case."""
    y = np.array([0, 0, 1, 1, 0, 1, 1, 0])
    yhat = np.array([0, 1, 1, 0, 0, 1, 0, 1])
    assert (
        -1 < informedness(y, yhat) < 1
    ), f"Expected a value between -1 and 1, got {informedness(y, yhat)}"
