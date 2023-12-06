import numpy as np
import pandas as pd
import polars as pl
import pytest
from typing import Union
from PredicTables.src.correlation_matrix import _correlation_matrix, \
                                                _highly_correlated_variables

@pytest.fixture
def correlated_data():
    np.random.seed(42)
    data = np.random.normal(size=(1000, 20))
    return data

@pytest.fixture
def pddf():
    return pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6], 'c': [7, 8, 9]})

@pytest.fixture
def pds():
    return pd.Series([1, 2, 3, 4, 5])

@pytest.fixture
def pldf():
    return pl.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6], 'c': [7, 8, 9]})

@pytest.fixture
def plser():
    return pl.Series([1, 2, 3, 4, 5])

@pytest.fixture
def pllf(pldf):
    return pldf.lazy()

@pytest.fixture
def arr():
    return np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])

@pytest.fixture
def lst():
    return [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]

@pytest.fixture
def corr():
    return np.array([[1., 1., 1.],
                     [1., 1., 1.],
                     [1., 1., 1.]])

@pytest.fixture
def corr_arr():
    return np.array([['x0', 'x1', 'x2'],
                     [1., 1., 1.],
                     [1., 1., 1.],
                     [1., 1., 1.]])                     

@pytest.fixture
def pddf_2(correlated_data):
    data = correlated_data[:, :10]
    return pd.DataFrame(data, columns=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'])

@pytest.fixture
def pds_2(correlated_data):
    data = correlated_data[:, 0]
    return pd.Series(data)

@pytest.fixture
def pldf_2(correlated_data):
    data = correlated_data[:, :10]
    df = pd.DataFrame(data, columns=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'])
    return pl.from_pandas(df)


@pytest.fixture
def plser_2(correlated_data):
    data = correlated_data[:, :10]
    df = pd.DataFrame(data, columns=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'])
    return pl.from_pandas(df['a'])

@pytest.fixture
def pllf_2(correlated_data):
    data = correlated_data[:, :10]
    df = pd.DataFrame(data, columns=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'])
    return pl.from_pandas(df).lazy()

@pytest.fixture
def arr_2(correlated_data):
    data = correlated_data[:, :10]
    df = pd.DataFrame(data, columns=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'])
    return df.to_numpy()

@pytest.fixture
def lst_2(correlated_data):
    data = correlated_data[:, :10]
    df = pd.DataFrame(data, columns=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'])
    return df.values.tolist()

@pytest.fixture
def pddf_corr_2(correlated_data):
    data = correlated_data[:, :10]
    df = pd.DataFrame(data, columns=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'])

@pytest.fixture
def pldf_corr_2():
    np.random.seed(42)
    data = np.random.normal(size=(100, 10))
    data[:, 1] = data[:, 0] * 2
    return pl.DataFrame(data, columns=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'])

@pytest.fixture
def pllf_corr_2():
    np.random.seed(42)
    data = np.random.normal(size=(100, 10))
    data[:, 1] = data[:, 0] * 2
    return pl.lazy.frame({'a': data[:, 0], 'b': data[:, 1], 'c': data[:, 2], 'd': data[:, 3], 'e': data[:, 4], 'f': data[:, 5], 'g': data[:, 6], 'h': data[:, 7], 'i': data[:, 8], 'j': data[:, 9]})

def test_correlation_matrix(pddf, pds, pldf, plser, pllf, arr, lst, corr, corr_arr):
    # Test pandas DataFrame
    assert isinstance(_correlation_matrix(pddf), pd.DataFrame)

    # # Test pandas Series
    # assert isinstance(_correlation_matrix(pds), pd.DataFrame)

    # Test polars DataFrame
    assert isinstance(_correlation_matrix(pldf), pd.DataFrame)

    # # Test polars Series
    # assert isinstance(_correlation_matrix(plser), pd.DataFrame)

    # Test polars LazyFrame
    assert isinstance(_correlation_matrix(pllf), pd.DataFrame)

    # # Test numpy array
    # assert isinstance(_correlation_matrix(arr), pd.DataFrame)

    # # Test list
    # assert isinstance(_correlation_matrix(lst), pd.DataFrame)

    # Test unsupported type (should raise TypeError)
    try:
        _correlation_matrix(123)
    except TypeError:
        pass
    else:
        assert False, "Expected TypeError"

    # Test method parameter
    assert isinstance(_correlation_matrix(pddf, method='pearson'), pd.DataFrame)
    assert isinstance(_correlation_matrix(pddf, method='pearson'), pd.DataFrame)
    assert isinstance(_correlation_matrix(pldf, method='pearson'), pd.DataFrame)
    assert isinstance(_correlation_matrix(pddf, method='pearson'), pd.DataFrame)
    assert isinstance(_correlation_matrix(pllf, method='pearson'), pd.DataFrame)
    # assert isinstance(_correlation_matrix(arr, method='pearson'), pd.DataFrame)
    # assert isinstance(_correlation_matrix(lst, method='pearson'), pd.DataFrame)

    # Test unsupported method (should raise ValueError)
    try:
        _correlation_matrix(pddf, method='unsupported')
    except ValueError:
        pass
    else:
        assert False, "Expected ValueError"

    # Test that the correct correlation matrix is calculated
    assert np.array_equal(_correlation_matrix(pddf), corr)
    # assert np.array_equal(_correlation_matrix(pds), corr)
    assert np.array_equal(_correlation_matrix(pldf), corr)
    # assert np.array_equal(_correlation_matrix(plser), corr)
    assert np.array_equal(_correlation_matrix(pllf), corr)
    # assert np.array_equal(_correlation_matrix(arr), corr_arr)
    # assert np.array_equal(_correlation_matrix(lst), corr_arr)

    # Test that the correlation matrix has the correct column names
    assert _correlation_matrix(pddf).columns.tolist() == ['a', 'b', 'c']


def test_highly_correlated_variables(pddf_2, pds_2, pldf_2, plser_2, pllf_2, arr_2, lst_2):
    # Test pandas DataFrame
    assert isinstance(_highly_correlated_variables(pddf_2), list)

    # # Test pandas Series
    # assert isinstance(_highly_correlated_variables(pds_2), list)

    # Test polars DataFrame
    assert isinstance(_highly_correlated_variables(pldf_2), list)

    # # Test polars Series
    # assert isinstance(_highly_correlated_variables(plser_2), list)

    # Test polars LazyFrame
    assert isinstance(_highly_correlated_variables(pllf_2), list)

    # # Test numpy array
    # assert isinstance(_highly_correlated_variables(arr_2), list)

    # # Test list
    # assert isinstance(_highly_correlated_variables(lst_2), list)

    # Test unsupported type (should raise TypeError)
    try:
        _highly_correlated_variables(123)
    except TypeError:
        pass
    else:
        assert False, "Expected TypeError"

    # Test method parameter
    assert isinstance(_highly_correlated_variables(pddf_2, method='pearson'), list)
    # assert isinstance(_highly_correlated_variables(pds_2, method='pearson'), list)
    assert isinstance(_highly_correlated_variables(pldf_2, method='pearson'), list)
    # assert isinstance(_highly_correlated_variables(plser_2, method='pearson'), list)
    assert isinstance(_highly_correlated_variables(pllf_2, method='pearson'), list)
    # assert isinstance(_highly_correlated_variables(arr_2, method='pearson'), list)
    # assert isinstance(_highly_correlated_variables(lst_2, method='pearson'), list)

    # Test threshold parameter
    assert isinstance(_highly_correlated_variables(pddf_2, threshold=0.5), list)
    # assert isinstance(_highly_correlated_variables(pds_2, threshold=0.5), list)
    assert isinstance(_highly_correlated_variables(pldf_2, threshold=0.5), list)
    # assert isinstance(_highly_correlated_variables(plser_2, threshold=0.5), list)
    assert isinstance(_highly_correlated_variables(pllf_2, threshold=0.5), list)
    # assert isinstance(_highly_correlated_variables(arr_2, threshold=0.5), list)
    # assert isinstance(_highly_correlated_variables(lst_2, threshold=0.5), list)
    
    # Test unsupported method (should raise ValueError)
    try:
        _highly_correlated_variables(pddf_2, method='unsupported')
    except ValueError:
        pass
    else:
        assert False, "Expected ValueError"