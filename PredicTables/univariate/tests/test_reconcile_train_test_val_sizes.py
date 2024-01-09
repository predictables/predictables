import pytest
import numpy as np
from PredicTables.univariate import reconcile_train_test_val_sizes


@pytest.mark.parametrize(
    "train_size, test_size, val_size, expected, round",
    [
        (None, None, None, (0.7, 0.2, 0.1), 1),
        (0.5, None, None, (0.5, 0.25, 0.25), 2),
        (None, 0.3, None, (0.35, 0.3, 0.35), 3),
        (None, None, 0.4, (0.3, 0.3, 0.4), 4),
        (0.6, 0.2, None, (0.6, 0.2, 0.2), 5),
        (0.6, None, 0.3, (0.6, 0.1, 0.3), 6),
        (None, 0.4, 0.1, (0.5, 0.4, 0.1), 7),
        (0.4, 0.4, 0.2, (0.4, 0.4, 0.2), 8),
    ],
)
def test_reconcile_train_test_val_sizes(
    train_size, test_size, val_size, expected, round
):
    result = [
        np.round(x, 2)
        for x in reconcile_train_test_val_sizes(train_size, test_size, val_size)
    ]

    assert (
        result[0] == expected[0]
    ), f"In round {round}, position 0, expected {expected[0]}, got {result[0]}"
    assert (
        result[1] == expected[1]
    ), f"In round {round}, position 1, expected {expected[1]}, got {result[1]}"
    assert (
        result[2] == expected[2]
    ), f"In round {round}, position 2, expected {expected[2]}, got {result[2]}"


def test_reconcile_train_test_val_sizes_error():
    with pytest.raises(ValueError):
        reconcile_train_test_val_sizes(0.4, 0.4, 0.3)
