import pytest
import numpy as np
import pandas as pd
import polars as pl

from scipy.stats import norm
from PredicTables.src.eigenvalue_analysis import _eigenvalues

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

@pytest.fixture
def expected():
    return pd.DataFrame({
        'feature': ['b', 'c', 'a'],
        'eigenvalue': [0, 0.186849, 2.813151],
        'log[condition_number]': [37.135003, 2.711757, 0.000000]
        })

def test_eigenvalues_pandas(pddf, expected):
    result = _eigenvalues(pddf)
    pd.testing.assert_frame_equal(result, expected, rtol=0.01, atol=0.01)

def test_eigenvalues_polars(pldf, expected):
    result = _eigenvalues(pldf)
    pd.testing.assert_frame_equal(result, expected, rtol=0.01, atol=0.01)

def test_eigenvalues_lazy(pllf, expected):
    result = _eigenvalues(pllf)
    pd.testing.assert_frame_equal(result, expected, rtol=0.01, atol=0.01)

def test_eigenvalues_numpy(arr, expected):
    result = _eigenvalues(arr)
    pd.testing.assert_frame_equal(result.iloc[:, 1:], expected.iloc[:, 1:], rtol=0.01, atol=0.01)

def test_eigenvalues_unsupported_type():
    with pytest.raises(TypeError):
        _eigenvalues(123)