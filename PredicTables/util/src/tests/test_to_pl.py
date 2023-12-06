# FILEPATH: /home/aweaver/work/hit-ratio-model/PredicTables/util/src/tests/test_to_pl.py

import pytest
import pandas as pd
import polars as pl
import numpy as np
from PredicTables.util import to_pl_df, to_pl_lf, to_pl_s


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
def df_input(request):
    if request.param == pd.DataFrame:
        return pd.DataFrame(np.random.rand(10, 3), columns=list("ABC"))
    elif request.param == pl.DataFrame:
        return pl.DataFrame(
            {"A": np.random.rand(10), "B": np.random.rand(10), "C": np.random.rand(10)}
        )
    elif request.param == pl.LazyFrame:
        return pl.DataFrame(
            {"A": np.random.rand(10), "B": np.random.rand(10), "C": np.random.rand(10)}
        ).lazy()
    elif request.param == pd.Series:
        return pd.Series(np.random.rand(10), name="A")
    elif request.param == pl.Series:
        return pl.Series("A", np.random.rand(10))
    elif request.param == np.ndarray:
        return np.random.rand(10, 3)
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
def lf_input(request):
    if request.param == pd.DataFrame:
        return pd.DataFrame(np.random.rand(10, 3), columns=list("ABC"))
    elif request.param == pl.DataFrame:
        return pl.DataFrame(
            {"A": np.random.rand(10), "B": np.random.rand(10), "C": np.random.rand(10)}
        )
    elif request.param == pl.LazyFrame:
        return pl.DataFrame(
            {"A": np.random.rand(10), "B": np.random.rand(10), "C": np.random.rand(10)}
        ).lazy()
    elif request.param == pd.Series:
        return pd.Series(np.random.rand(10), name="A")
    elif request.param == pl.Series:
        return pl.Series("A", np.random.rand(10))
    elif request.param == np.ndarray:
        return np.random.rand(10, 3)
    else:
        return "invalid"


@pytest.fixture(params=[pd.Series, pl.Series, np.ndarray, str])
def series_input(request):
    if request.param == pd.Series:
        return pd.Series(np.random.rand(10))
    elif request.param == pl.Series:
        return pl.Series("A", np.random.rand(10))
    elif request.param == np.ndarray:
        return np.random.rand(10)
    else:
        return "invalid"


def test_to_pl_df(df_input):
    if isinstance(df_input, str):
        with pytest.raises(TypeError):
            to_pl_df(df_input)
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
                result.shape[1] == 3
            ), f"DataFrame should have three columns for non-Series input, but has {result.shape[1]}"


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


def test_to_pl_s(series_input):
    if isinstance(series_input, str):
        with pytest.raises(TypeError):
            to_pl_s(series_input)
    else:
        result = to_pl_s(series_input)
        assert isinstance(
            result, pl.Series
        ), f"Expected output type is pl.Series, got {type(result)}"
        assert (
            result.shape[0] == 10
        ), f"Series should have 10 elements, but has {result.shape[0]}"


def test_to_pl_s_ndarray_2d():
    ndarray_2d = np.random.rand(10, 2)
    with pytest.raises(ValueError):
        to_pl_s(ndarray_2d)
