from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
import pytest

from predictables.util import to_pl_df, to_pl_lf, to_pl_s


@pytest.fixture(
    params=[
        pd.DataFrame,
        pl.DataFrame,
        pl.LazyFrame,
        pd.Series,
        pl.Series,
        np.ndarray,
        str,
    ]
)
def df_input(
    request: pytest.FixtureRequest,
) -> (
    pd.DataFrame
    | pl.DataFrame
    | pl.LazyFrame
    | pd.Series
    | pl.Series
    | np.ndarray
    | str
):
    rg = np.random.Generator(np.random.PCG64(12345))
    if request.param == pd.DataFrame:
        return pd.DataFrame(rg.random((10, 3)), columns=list("ABC"))
    elif request.param in [pl.DataFrame, pl.LazyFrame]:
        df = pl.DataFrame({"A": rg.random(10), "B": rg.random(10), "C": rg.random(10)})
        return df.lazy() if request.param == pl.LazyFrame else df
    elif request.param == pd.Series:
        return pd.Series(rg.random(10), name="A")
    elif request.param == pl.Series:
        return pl.Series("A", rg.random(10))
    elif request.param == np.ndarray:
        return rg.random((10, 3))
    else:
        return "invalid"


@pytest.fixture(
    params=[
        pd.DataFrame,
        pl.DataFrame,
        pl.LazyFrame,
        pd.Series,
        pl.Series,
        np.ndarray,
        str,
    ]
)
def lf_input(
    request: pytest.FixtureRequest,
) -> (
    pd.DataFrame
    | pl.DataFrame
    | pl.LazyFrame
    | pd.Series
    | pl.Series
    | np.ndarray
    | str
):
    rg = np.random.Generator(np.random.PCG64(12345))
    if request.param == pd.DataFrame:
        return pd.DataFrame(rg.random((10, 3)), columns=list("ABC"))
    elif request.param in [pl.DataFrame, pl.LazyFrame]:
        df = pl.DataFrame({"A": rg.random(10), "B": rg.random(10), "C": rg.random(10)})
        return df.lazy() if request.param == pl.LazyFrame else df
    elif request.param == pd.Series:
        return pd.Series(rg.random(10), name="A")
    elif request.param == pl.Series:
        return pl.Series("A", rg.random(10))
    elif request.param == np.ndarray:
        return rg.random((10, 3))
    else:
        return "invalid"


@pytest.fixture(params=[pd.Series, pl.Series, np.ndarray, str])
def series_input(
    request: pytest.FixtureRequest,
) -> pd.Series | pl.Series | np.ndarray | str:
    rg = np.random.Generator(np.random.PCG64(12345))
    if request.param == pd.Series:
        return pd.Series(rg.random(10))
    elif request.param == pl.Series:
        return pl.Series("A", rg.random(10))
    elif request.param == np.ndarray:
        return rg.random(10)
    else:
        return "invalid"


def test_to_pl_df(df_input: pd.DataFrame | pl.DataFrame | pl.LazyFrame | str) -> None:
    expected_columns = 3

    if isinstance(df_input, str):
        with pytest.raises(TypeError):
            to_pl_df(df_input) # type: ignore[arg-type]
    else:
        result = to_pl_df(df_input)
        assert isinstance(
            result, pl.DataFrame
        ), f"Expected output type is pl.DataFrame, got {type(result)}"
        if isinstance(df_input, (pd.Series, pl.Series)):
            assert (
                result.shape[1] == 1
            ), f"DataFrame should have only one column for Series input, but has {result.shape[1]}"
        else:
            assert (
                result.shape[1] == expected_columns
            ), f"DataFrame should have {expected_columns} columns for non-Series input, but has {result.shape[1]}"


def test_to_pl_lf(lf_input):
    if isinstance(lf_input, str):
        with pytest.raises(TypeError):
            to_pl_lf(lf_input)
    else:
        result = to_pl_lf(lf_input)
        assert isinstance(
            result, pl.LazyFrame
        ), f"Expected output type is pl.LazyFrame, got {type(result)}"
        if isinstance(lf_input, (pd.Series, pl.Series)):
            assert (
                result.collect().shape[1] == 1
            ), f"LazyFrame should have only one column for Series input, but has {result.collect().shape[1]}"
        else:
            assert (
                result.collect().shape[1] == 3
            ), f"LazyFrame should have three columns for non-Series input, but has {result.collect().shape[1]}"


def test_to_pl_s(series_input: pd.Series | pl.Series | np.ndarray | str) -> None:
    expected_shape = 10
    if isinstance(series_input, str):
        with pytest.raises(TypeError):
            to_pl_s(series_input)
    else:
        result = to_pl_s(series_input)
        assert isinstance(
            result, pl.Series
        ), f"Expected output type is pl.Series, got {type(result)}"
        assert (
            result.shape[0] == expected_shape
        ), f"Series should have 10 elements, but has {result.shape[0]}"


def test_to_pl_s_ndarray_2d() -> None:
    rg = np.random.Generator(np.random.PCG64(12345))
    ndarray_2d = rg.random(10, 2)
    with pytest.raises(ValueError):
        to_pl_s(ndarray_2d)
