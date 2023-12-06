import pytest
import numpy as np
import pandas as pd
import polars as pl
from scipy.stats import norm
from PredicTables.src.variance_inflation_factor import _vif_i, _vif

# Create test data
@pytest.fixture
def a():
    # Create random variables with random seed=42
    rvs = pd.Series(norm.rvs(size=3, random_state=42))
    return pd.Series([1, 2, 3]) * rvs

@pytest.fixture
def b():
    # Create random variables with random seed=42
    rvs = pd.Series(norm.rvs(size=3, random_state=421))
    return pd.Series([4, 5, 6]) * rvs

@pytest.fixture
def c():
    # Create random variables with random seed=42
    rvs = pd.Series(norm.rvs(size=3, random_state=422))
    return pd.Series([7, 8, 9]) * rvs


@pytest.fixture
def pddf(a, b, c):
    return pd.DataFrame({'a': a, 'b': b, 'c': c})

@pytest.fixture
def pldf(a, b, c):
    return pl.DataFrame({'a': a, 'b': b, 'c': c})

@pytest.fixture
def pllf(pldf):
    return pldf.lazy()

@pytest.fixture
def arr(a, b, c):
    return np.array([a.tolist(), b.tolist(), c.tolist()]).T

def test_vif_i(pddf, pldf, pllf, arr):
    # Test pandas DataFrame
    assert _vif_i(pddf, 'a') == pytest.approx(1.5, 0.1)
    assert _vif_i(pddf, 'b') == pytest.approx(5.8, 0.1)
    assert _vif_i(pddf, 'c') == pytest.approx(6.8, 0.1)

    # Test polars DataFrame
    assert _vif_i(pldf, 'a') == pytest.approx(1.5, 0.1)
    assert _vif_i(pldf, 'b') == pytest.approx(5.8, 0.1)
    assert _vif_i(pldf, 'c') == pytest.approx(6.8, 0.1)

    # Test polars LazyFrame
    assert _vif_i(pllf, 'a') == pytest.approx(1.5, 0.1)
    assert _vif_i(pllf, 'b') == pytest.approx(5.8, 0.1)
    assert _vif_i(pllf, 'c') == pytest.approx(6.8, 0.1)

    # Test numpy array
    assert _vif_i(arr, 0) == pytest.approx(1.5, 0.1)
    assert _vif_i(arr, 1) == pytest.approx(5.8, 0.1)
    assert _vif_i(arr, 2) == pytest.approx(6.8, 0.1)

    # Test unsupported type
    with pytest.raises(TypeError):
        _vif_i(123, 'a')

def test_vif_pandas(pddf):
    expected = pd.DataFrame({'feature': ['c', 'b', 'a'], 'vif_score': [6.8, 5.8, 1.5]})
    result = _vif(pddf, show_progress=False)
    pd.testing.assert_frame_equal(result, expected, rtol=0.1, atol=0.1)

def test_vif_polars(pldf):
    expected = pd.DataFrame({'feature': ['c', 'b', 'a'], 'vif_score': [6.8, 5.8, 1.5]})
    result = _vif(pldf, show_progress=False)
    pd.testing.assert_frame_equal(result, expected, rtol=0.1, atol=0.1)

def test_vif_lazy(pllf):
    expected = pd.DataFrame({'feature': ['c', 'b', 'a'], 'vif_score': [6.8, 5.8, 1.5]})
    result = _vif(pllf, show_progress=False)
    pd.testing.assert_frame_equal(result, expected, rtol=0.1, atol=0.1)

def test_vif_numpy(arr):
    expected = pd.DataFrame({'feature': [2, 1, 0], 'vif_score': [6.8, 5.8, 1.5]})
    result = _vif(arr, show_progress=False)
    pd.testing.assert_frame_equal(result, expected, rtol=0.1, atol=0.1)

def test_vif_unsupported_type():
    with pytest.raises(TypeError):
        _vif(123, show_progress=False)