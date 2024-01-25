import numpy as np
import pandas as pd
import polars as pl
import pytest

from predictables.util import is_all_none
from predictables.util.src._is_all_none import (
    _is_all_none_np,
    _is_all_none_pd,
    _is_all_none_pl,
)

# FILEPATH: /home/aweaver/work/hit-ratio-model/PredicTables/util/src/tests/test_is_all_none.py


@pytest.fixture(params=[pd.Series, pl.Series, str])
def series_input(request):
    if request.param == pd.Series:
        return pd.Series([None, None, None])
    elif request.param == pl.Series:
        return pl.Series("A", [None, None, None])
    else:
        return "invalid"


def test_is_all_none_pd(series_input):
    if isinstance(series_input, str):
        with pytest.raises(TypeError):
            _is_all_none_pd(series_input)
    else:
        result = _is_all_none_pd(series_input)
        assert (
            result
        ), f"Expected True (eg all None), got {result} for {series_input}, with input type {type(series_input)}"


def test_is_all_none_pd_not_all_none():
    series = pd.Series([1, None, 3])
    result = _is_all_none_pd(series)
    assert (
        not result
    ), f"Expected False (eg not all None), got {result} for {series}, with input type {type(series)}"
    result2 = is_all_none(series)
    assert (
        not result2
    ), f"Expected False (eg not all None), got {result2} for {series}, with input type {type(series)}"


def test_is_all_none_pl(series_input):
    if isinstance(series_input, str):
        with pytest.raises(TypeError):
            _is_all_none_pl(series_input)
    else:
        result = _is_all_none_pl(series_input)
        assert (
            result
        ), f"Expected True (eg all None), got {result} for {series_input}, with input type {type(series_input)}"


def test_is_all_none_pl_not_all_none():
    series = pl.Series("A", [1, None, 3])
    result = _is_all_none_pl(series)
    assert (
        not result
    ), f"Expected False (eg not all None), got {result} for {series}, with input type {type(series)}"
    result2 = is_all_none(series)
    assert (
        not result2
    ), f"Expected False (eg not all None), got {result2} for {series}, with input type {type(series)}"


# def test_is_all_none_np(series_input):
#     if isinstance(series_input, str):
#         with pytest.raises(TypeError):
#             _is_all_none_np(series_input)
#     else:
#         result = _is_all_none_np(series_input)
#         assert result, f"Expected True (eg all None), got {result} for {series_input}, with input type {type(series_input)}"


def test_is_all_none_np_not_all_none():
    series = np.array([1, None, 3])
    result = _is_all_none_np(series)
    assert (
        not result
    ), f"Expected False (eg not all None), got {result} for {series}, with input type {type(series)}"


def test_is_all_none(series_input):
    if isinstance(series_input, np.ndarray):
        pass
    else:
        if isinstance(series_input, str):
            with pytest.raises(TypeError):
                is_all_none(series_input)
        else:
            result = is_all_none(series_input)
            assert (
                result
            ), f"Expected True (eg all None), got {result} for {series_input}, with input type {type(series_input)}"
