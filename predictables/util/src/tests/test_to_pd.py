from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
import pytest

from predictables.util import to_pd_df, to_pd_s


@pytest.fixture(params=[pd.Series, pl.Series, np.ndarray])
def series_input(
    request: pytest.FixtureRequest,
) -> pd.Series | pl.Series | np.ndarray | str:
    if request.param == pd.Series:
        return pd.Series([1, 2, 3, 4, 5])
    elif request.param == pl.Series:
        return pl.Series("a", [1, 2, 3])
    elif request.param == np.ndarray:
        return np.array([1, 2, 3, 4, 5])
    return "invalid"


@pytest.fixture(
    params=[pd.DataFrame, pl.DataFrame, pl.LazyFrame, pd.Series, pl.Series, np.ndarray]
)
def df_input(
    request: pytest.FixtureRequest,
) -> (
    pd.DataFrame
    | pl.DataFrame
    | pl.LazyFrame
    | np.ndarray
    | pd.Series
    | pl.Series
    | str
):
    rg = np.random.Generator(np.random.PCG64(12345))
    if request.param == pd.DataFrame:
        return pd.DataFrame(rg.random((5, 5)), columns=list("abcde"))
    elif request.param in [pl.DataFrame, pl.LazyFrame]:
        out = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        return out.lazy() if request.param == pl.LazyFrame else out
    elif request.param == pd.Series:
        return pd.Series([1, 2, 3, 4, 5])
    elif request.param == pl.Series:
        return pl.Series("a", [1, 2, 3])
    elif request.param == np.ndarray:
        return np.array([[1, 2, 3], [4, 5, 6]])
    return "invalid"


def test_to_pd_df(
    df_input: pd.DataFrame | pl.DataFrame | pl.LazyFrame | np.ndarray,
) -> None:
    result = to_pd_df(df_input)
    assert isinstance(
        result, pd.DataFrame
    ), f"Expected pd.DataFrame, got {type(result)}"


def test_to_pd_df_unsupported_type() -> None:
    with pytest.raises(TypeError):
        to_pd_df("unsupported type")


def test_to_pd_s(series_input: pd.Series | pl.Series | np.ndarray | str) -> None:
    result = to_pd_s(series_input)
    assert isinstance(result, pd.Series), f"Expected pd.Series, got {type(result)}"


def test_to_pd_s_unsupported_type() -> None:
    with pytest.raises(TypeError):
        to_pd_s("unsupported type")


def test_to_pd_s_empty_series() -> None:
    empty_series = pd.Series([])
    result = to_pd_s(empty_series)
    assert result.empty, "Expected an empty pd.Series"
