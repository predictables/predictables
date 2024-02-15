# FILEPATH: /home/aweaver/work/hit-ratio-model/PredicTables/util/tests/test_to_pd.py

import numpy as np
import pandas as pd  # type: ignore
import polars as pl  # type: ignore
import pytest

from predictables.util import to_pd_df, to_pd_s


@pytest.fixture(params=[pd.Series, pl.Series, np.ndarray])
def series_input(request):
    if request.param == pd.Series:
        return pd.Series([1, 2, 3, 4, 5])
    elif request.param == pl.Series:
        return pl.Series("a", [1, 2, 3])
    elif request.param == np.ndarray:
        return np.array([1, 2, 3, 4, 5])


@pytest.fixture(
    params=[
        pd.DataFrame,
        pl.DataFrame,
        pl.LazyFrame,
        pd.Series,
        pl.Series,
        np.ndarray,
    ]
)
def df_input(request):
    if request.param == pd.DataFrame:
        return pd.DataFrame(np.random.rand(5, 5), columns=list("abcde"))
    elif request.param == pl.DataFrame:
        return pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    elif request.param == pl.LazyFrame:
        return pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}).lazy()
    elif request.param == pd.Series:
        return pd.Series([1, 2, 3, 4, 5])
    elif request.param == pl.Series:
        return pl.Series("a", [1, 2, 3])
    elif request.param == np.ndarray:
        return np.array([[1, 2, 3], [4, 5, 6]])


def test_to_pd_df(df_input):
    result = to_pd_df(df_input)
    assert isinstance(
        result, pd.DataFrame
    ), f"Expected pd.DataFrame, got {type(result)}"


def test_to_pd_df_unsupported_type():
    with pytest.raises(TypeError):
        to_pd_df("unsupported type")


def test_to_pd_s(series_input):
    result = to_pd_s(series_input)
    assert isinstance(
        result, pd.Series
    ), f"Expected pd.Series, got {type(result)}"


def test_to_pd_s_unsupported_type():
    with pytest.raises(TypeError):
        to_pd_s("unsupported type")


def test_to_pd_s_empty_series():
    empty_series = pd.Series([])
    result = to_pd_s(empty_series)
    assert result.empty, "Expected an empty pd.Series"
