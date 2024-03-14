import numpy as np
import pandas as pd
import polars as pl
import pytest

from predictables.src._utils import _to_numpy, _to_pandas, _to_polars


# Create test data
@pytest.fixture
def pddf():
    return pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})


@pytest.fixture
def pds():
    return pd.Series([1, 2, 3])


@pytest.fixture
def pldf():
    return pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})


@pytest.fixture
def plser():
    return pl.Series([1, 2, 3])


@pytest.fixture
def pllf(pldf):
    return pldf.lazy()


@pytest.fixture
def arr():
    return np.array([1, 2, 3])


@pytest.fixture
def lst():
    return [1, 2, 3]


def test_to_numpy(pddf, pds, pldf, plser, pllf, arr, lst):
    # Test pandas DataFrame
    assert np.array_equal(_to_numpy(pddf), pddf.to_numpy())

    # Test pandas Series
    assert np.array_equal(_to_numpy(pds), pds.to_numpy())

    # Test polars DataFrame
    assert np.array_equal(_to_numpy(pldf), pldf.to_numpy())

    # Test polars Series
    assert np.array_equal(_to_numpy(plser), plser.to_numpy())

    # Test polars LazyFrame
    assert np.array_equal(_to_numpy(pllf), pllf.collect().to_numpy())

    # Test numpy array
    assert np.array_equal(_to_numpy(arr), arr)

    # Test list
    assert np.array_equal(_to_numpy(lst), np.array(lst))

    # Test unsupported type
    try:
        _to_numpy(123)
    except TypeError:
        pass
    else:
        raise AssertionError("Expected TypeError")


def test_to_polars(pddf, pds, pldf, plser, pllf, arr, lst):
    # Test pandas DataFrame to polars DataFrame
    assert isinstance(_to_polars(pddf, to="dataframe"), pl.DataFrame)

    # Test pandas DataFrame to polars LazyFrame
    assert isinstance(_to_polars(pddf, to="lazyframe"), pl.LazyFrame)

    # Test pandas DataFrame to polars Series (should raise ValueError)
    try:
        _to_polars(pddf, to="series")
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError")

    # Test pandas Series to polars DataFrame
    assert isinstance(_to_polars(pds, to="dataframe"), pl.DataFrame)

    # Test pandas Series to polars Series
    assert isinstance(_to_polars(pds, to="series"), pl.Series)

    # Test pandas Series to polars LazyFrame
    assert isinstance(_to_polars(pds, to="lazyframe"), pl.LazyFrame)

    # Test polars DataFrame to polars DataFrame
    assert isinstance(_to_polars(pldf, to="dataframe"), pl.DataFrame)

    # Test polars DataFrame to polars LazyFrame
    assert isinstance(_to_polars(pldf, to="lazyframe"), pl.LazyFrame)

    # Test polars DataFrame to polars Series (should raise ValueError)
    try:
        _to_polars(pldf, to="series")
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError")

    # Test polars Series to polars DataFrame
    assert isinstance(_to_polars(plser, to="dataframe"), pl.DataFrame)

    # Test polars Series to polars Series
    assert isinstance(_to_polars(plser, to="series"), pl.Series)

    # Test polars Series to polars LazyFrame
    assert isinstance(_to_polars(plser, to="lazyframe"), pl.LazyFrame)

    # Test polars LazyFrame to polars DataFrame
    assert isinstance(_to_polars(pllf, to="dataframe"), pl.DataFrame)

    # Test polars LazyFrame to polars LazyFrame
    assert isinstance(_to_polars(pllf, to="lazyframe"), pl.LazyFrame)

    # Test polars LazyFrame to polars Series (should raise ValueError)
    try:
        _to_polars(pllf, to="series")
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError")

    # Test numpy array to polars DataFrame
    assert isinstance(_to_polars(arr, to="dataframe"), pl.DataFrame)

    # Test numpy array to polars LazyFrame
    assert isinstance(_to_polars(arr, to="lazyframe"), pl.LazyFrame)

    # Test numpy array to polars Series (should raise ValueError)
    try:
        _to_polars(arr, to="series")
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError")

    # Test list to polars DataFrame
    assert isinstance(_to_polars(lst, to="dataframe"), pl.DataFrame)

    # Test list to polars LazyFrame
    assert isinstance(_to_polars(lst, to="lazyframe"), pl.LazyFrame)

    # Test list to polars Series (should raise ValueError)
    try:
        _to_polars(lst, to="series")
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError")

    # Test unsupported type (should raise TypeError)
    try:
        _to_polars(123)
    except TypeError:
        pass
    else:
        raise AssertionError("Expected TypeError")


def test_to_pandas(pddf, pds, pldf, plser, pllf, arr, lst):
    # Test pandas DataFrame to pandas DataFrame
    assert isinstance(_to_pandas(pddf, to="dataframe"), pd.DataFrame)

    # Test pandas DataFrame to pandas Series (should raise ValueError)
    try:
        _to_pandas(pddf, to="series")
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError")

    # Test pandas Series to pandas DataFrame
    assert isinstance(_to_pandas(pds, to="dataframe"), pd.DataFrame)

    # Test pandas Series to pandas Series
    assert isinstance(_to_pandas(pds, to="series"), pd.Series)

    # Test polars DataFrame to pandas DataFrame
    assert isinstance(_to_pandas(pldf, to="dataframe"), pd.DataFrame)

    # Test polars DataFrame to pandas Series (should raise ValueError)
    try:
        _to_pandas(pldf, to="series")
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError")

    # Test polars Series to pandas DataFrame
    assert isinstance(_to_pandas(plser, to="dataframe"), pd.DataFrame)

    # Test polars Series to pandas Series
    assert isinstance(_to_pandas(plser, to="series"), pd.Series)

    # Test polars LazyFrame to pandas DataFrame
    assert isinstance(_to_pandas(pllf, to="dataframe"), pd.DataFrame)

    # Test polars LazyFrame to pandas Series (should raise ValueError)
    try:
        _to_pandas(pllf, to="series")
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError")

    # Test numpy array to pandas DataFrame
    assert isinstance(_to_pandas(arr, to="dataframe"), pd.DataFrame)

    # Test numpy array to pandas Series (should raise ValueError)
    try:
        _to_pandas(arr, to="series")
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError")

    # Test list to pandas DataFrame
    assert isinstance(_to_pandas(lst, to="dataframe"), pd.DataFrame)

    # Test list to pandas Series (should raise ValueError)
    try:
        _to_pandas(lst, to="series")
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError")

    # Test unsupported type (should raise TypeError)
    try:
        _to_pandas(123)
    except TypeError:
        pass
    else:
        raise AssertionError("Expected TypeError")


# Create test data
@pytest.fixture
def pddf_bincol():
    return pd.DataFrame(
        {"a": ["0", "1", "999"], "b": ["0", "1", "2"], "c": ["0", "1", "2", "3"]}
    )


@pytest.fixture
def pldf_bincol():
    return pl.DataFrame(
        {"a": ["0", "1", "999"], "b": ["0", "1", "2"], "c": ["0", "1", "2", "3"]}
    )


@pytest.fixture
def pllf_bincol(pldf_bincol):
    return pldf_bincol.lazy()