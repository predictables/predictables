import pytest
import pandas as pd
from pandas.testing import (
    assert_frame_equal as pd_assert_frame_equal,
    assert_series_equal as pd_assert_series_equal,
)
import polars as pl
from polars.testing import (
    assert_frame_equal as pl_assert_frame_equal,
    assert_series_equal as pl_assert_series_equal,
)
from typing import Union
from util import to_pd_df, to_pl_df, to_pl_lf, to_pl_s


@pytest.fixture
def pandas_df():
    return pd.DataFrame({"A": [1, 2], "B": [3, 4]})


@pytest.fixture
def polars_df():
    return pl.DataFrame({"A": pl.Series([1, 2]), "B": pl.Series([3, 4])})


@pytest.fixture
def polars_lazy(polars_df):
    return polars_df.lazy()


@pytest.fixture
def pd_series():
    return pd.Series([1, 2, 3])


@pytest.fixture
def pl_series(pd_series):
    return pl.Series(pd_series)


################### Test to_pd_df #############################################


# Test the conversion from pd.DataFrame -> pd.DataFrame
def test_pddf_from_pandas_df(pandas_df):
    assert isinstance(
        to_pd_df(pandas_df), pd.DataFrame
    ), f"to_pd_df({type(pandas_df)}) is not a pd.DataFrame"
    pd_assert_frame_equal(to_pd_df(pandas_df), pandas_df)


# Test the conversion from pl.DataFrame -> pd.DataFrame
def test_pddf_from_polars_df(polars_df):
    assert isinstance(
        to_pd_df(polars_df), pd.DataFrame
    ), f"to_pd_df({type(polars_df)}) is not a pd.DataFrame"
    pd_assert_frame_equal(to_pd_df(polars_df), polars_df.to_pandas())


# Test the conversion from pl.LazyFrame -> pd.DataFrame
def test_pddf_from_polars_lazy(polars_lazy):
    assert isinstance(
        to_pd_df(polars_lazy), pd.DataFrame
    ), f"to_pd_df({type(polars_lazy)}) is not a pd.DataFrame"
    pd_assert_frame_equal(to_pd_df(polars_lazy), polars_lazy.collect().to_pandas())


# Test the conversion from pd.Series -> pd.DataFrame
def test_pddf_from_pandas_series():
    s = pd.Series([1, 2])
    assert isinstance(
        to_pd_df(s), pd.DataFrame
    ), f"to_pd_df({type(s)}) is not a pd.DataFrame"
    pd_assert_frame_equal(to_pd_df(s), s.to_frame())


# Test the conversion from pl.Series -> pd.DataFrame
def test_pddf_from_polars_series():
    s = pl.Series([1, 2])
    assert isinstance(
        to_pd_df(s), pd.DataFrame
    ), f"to_pd_df({type(s)}) is not a pd.DataFrame"
    pd_assert_frame_equal(to_pd_df(s), s.to_pandas().to_frame())


# Test that there is no output from the conversion from invalid input
def test_pddf_from_invalid_input_is_error():
    with pytest.raises(TypeError):
        to_pd_df("invalid input")


################### Test to_pl_df #############################################


def test_pldf_from_pandas_df(pandas_df):
    assert isinstance(
        to_pl_df(pandas_df), pl.DataFrame
    ), f"to_pl_df({type(pandas_df)}) is not a pl.DataFrame"
    pl_assert_frame_equal(to_pl_df(pandas_df), pl.from_pandas(pandas_df))


def test_pldf_from_polars_df(polars_df):
    assert isinstance(
        to_pl_df(polars_df), pl.DataFrame
    ), f"to_pl_df({type(polars_df)}) is not a pl.DataFrame"
    pl_assert_frame_equal(to_pl_df(polars_df), polars_df)


def test_pldf_from_polars_lazy(polars_lazy):
    assert isinstance(
        to_pl_df(polars_lazy), pl.DataFrame
    ), f"to_pl_df({type(polars_lazy)}) is not a pl.DataFrame"
    pl_assert_frame_equal(to_pl_df(polars_lazy), polars_lazy.collect())


def test_pldf_from_pandas_series():
    s = pd.Series([1, 2])
    assert isinstance(
        to_pl_df(s), pl.DataFrame
    ), f"to_pl_df({type(s)}) is not a pl.DataFrame"
    pl_assert_frame_equal(to_pl_df(s), pl.from_pandas(s.to_frame()))


def test_pldf_from_polars_series():
    s = pl.Series([1, 2])
    assert isinstance(
        to_pl_df(s), pl.DataFrame
    ), f"to_pl_df({type(s)}) is not a pl.DataFrame"
    pl_assert_series_equal(to_pl_df(s)[s.name], s)


def test_pldf_from_invalid_input_is_error():
    with pytest.raises(TypeError):
        to_pl_df("invalid input")


################### Test to_pl_lf #############################################


def test_pllf_from_pandas_df(pandas_df):
    assert isinstance(
        to_pl_lf(pandas_df), pl.LazyFrame
    ), f"to_pl_lf({type(pandas_df)}) is not a pl.LazyFrame"
    pl_assert_frame_equal(to_pl_lf(pandas_df).collect(), pl.from_pandas(pandas_df))


def test_pllf_from_polars_df(polars_df):
    assert isinstance(
        to_pl_lf(polars_df), pl.LazyFrame
    ), f"to_pl_lf({type(polars_df)}) is not a pl.LazyFrame"
    pl_assert_frame_equal(to_pl_lf(polars_df).collect(), polars_df)


def test_pllf_from_polars_lazy(polars_lazy):
    assert isinstance(
        to_pl_lf(polars_lazy), pl.LazyFrame
    ), f"to_pl_lf({type(polars_lazy)}) is not a pl.LazyFrame"
    pl_assert_frame_equal(to_pl_lf(polars_lazy).collect(), polars_lazy.collect())


def test_pllf_from_pandas_series():
    s = pd.Series([1, 2])
    assert isinstance(
        to_pl_lf(s), pl.LazyFrame
    ), f"to_pl_lf({type(s)}) is not a pl.LazyFrame"
    pl_assert_frame_equal(to_pl_lf(s).collect(), pl.from_pandas(s.to_frame()))


def test_pllf_from_polars_series():
    s = pl.Series([1, 2])
    assert isinstance(
        to_pl_lf(s), pl.LazyFrame
    ), f"to_pl_lf({type(s)}) is not a pl.LazyFrame"
    pl_assert_series_equal(to_pl_lf(s).collect()[s.name], s)


def test_pllf_from_invalid_input_is_error():
    with pytest.raises(TypeError):
        to_pl_lf("invalid input")


################### Test to_pl_s #############################################


def test_to_pl_s_with_pd_series(pd_series, pl_series):
    result = to_pl_s(pd_series)
    assert isinstance(
        result, pl.Series
    ), f"to_pl_s({type(pd_series)}) is not a pl.Series"
    assert (
        len(result) == len(pd_series)
    ), f"to_pl_s({type(pd_series)}) has length {len(result)} instead of {len(pd_series)}"
    assert (
        result.shape == pd_series.shape
    ), f"to_pl_s({type(pd_series)}) has shape {result.shape} instead of {pd_series.shape}"
    assert (
        result.dtype == pl_series.dtype
    ), f"to_pl_s({result}) has dtype {result.dtype} instead of {pl_series.dtype}"


def test_to_pl_s_with_pl_series(pl_series):
    result = to_pl_s(pl_series)
    assert isinstance(
        result, pl.Series
    ), f"to_pl_s({type(pl_series)}) is not a pl.Series"
    assert (
        len(result) == len(pl_series)
    ), f"to_pl_s({type(pl_series)}) has length {len(result)} instead of {len(pl_series)}"
    assert (
        result.shape == pl_series.shape
    ), f"to_pl_s({type(pl_series)}) has shape {result.shape} instead of {pl_series.shape}"
    assert (
        result.dtype == pl_series.dtype
    ), f"to_pl_s({type(pl_series)}) has dtype {result.dtype} instead of {pl_series.dtype}"


def test_to_pl_s_with_empty_series():
    s = pd.Series([])
    with pytest.raises(TypeError):
        to_pl_s(s)


def test_to_pl_s_with_single_value_series():
    s = pd.Series([1])
    result = to_pl_s(s)
    assert isinstance(result, pl.Series), f"to_pl_s({type(s)}) is not a pl.Series"
    assert len(result) == 1, f"to_pl_s({type(s)}) has length {len(result)} instead of 1"
    assert result[0] == 1, f"to_pl_s({type(s)}) has value {result[0]} instead of 1"


def test_to_pl_s_with_invalid_input_type():
    with pytest.raises(AttributeError):
        to_pl_s(123)
