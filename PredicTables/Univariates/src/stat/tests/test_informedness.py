import numpy as np
from ..informedness import informedness


def test_informedness_all_zeros():
    y = np.zeros(10)
    yhat = np.zeros(10)
    assert informedness(y, yhat) == -1, f"Expected -1, got {informedness(y, yhat)}"


def test_informedness_perfect_classifier():
    y = np.array([0, 0, 1, 1])
    yhat = np.array([0, 0, 1, 1])
    assert informedness(y, yhat) == 1, f"Expected 1, got {informedness(y, yhat)}"


def test_informedness_completely_wrong_classifier():
    y = np.array([0, 0, 1, 1])
    yhat = np.array([1, 1, 0, 0])
    assert informedness(y, yhat) == -1, f"Expected -1, got {informedness(y, yhat)}"


def test_informedness_general_case():
    y = np.array([0, 0, 1, 1, 0, 1, 1, 0])
    yhat = np.array([0, 1, 1, 0, 0, 1, 0, 1])
    assert (
        -1 < informedness(y, yhat) < 1
    ), f"Expected a value between -1 and 1, got {informedness(y, yhat)}"
