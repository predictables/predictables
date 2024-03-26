import pytest
from predictables.encoding.src.lagged_mean_encoding.sum.validation import (
    validate_offset,
)


# Dummy function to apply the decorator
@validate_offset
def dummy_func_with_offset(offset=0):
    return offset


# Test cases
def test_validate_offset_positive():
    assert dummy_func_with_offset(offset=10) == 10


def test_validate_offset_default():
    assert dummy_func_with_offset() == 0


def test_validate_offset_negative():
    with pytest.raises(ValueError):
        dummy_func_with_offset(offset=-1)
