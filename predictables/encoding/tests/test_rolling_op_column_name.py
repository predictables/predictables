from predictables.encoding.src.lagged_mean_encoding._rolling_op_column_name import (
    rolling_op_column_name,
)
import pytest


@pytest.mark.parametrize(
    "op, value_col, category_col, offset, window, expected",
    [
        ("ROLLING_SUM", "value", None, 0, 0, "ROLLING_SUM(value[ALL])[lag:0/win:0]"),
        (
            "ROLLING_MEAN",
            "value",
            "category",
            10,
            20,
            "ROLLING_MEAN(value[category])[lag:10/win:20]",
        ),
        (
            "ROLLING_SUM",
            "value",
            "category",
            0,
            0,
            "ROLLING_SUM(value[category])[lag:0/win:0]",
        ),
        (
            "ROLLING_MEAN",
            "value",
            None,
            10,
            20,
            "ROLLING_MEAN(value[ALL])[lag:10/win:20]",
        ),
    ],
)
def test_rolling_op_column_name(op, value_col, category_col, offset, window, expected):
    assert (
        rolling_op_column_name(op, value_col, category_col, offset, window) == expected
    ), f"Expected {expected} but got {rolling_op_column_name(op, value_col, category_col, offset, window)}."
