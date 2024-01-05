import pandas as pd
import numpy as np
import pytest
import itertools

from scipy.stats import pearsonr
from pandas.testing import assert_frame_equal

from PredicTables.corr.src.cts_cts import calc_continuous_continuous_corr


@pytest.fixture
def df_corr_neg1():
    """Create a synthetic dataset with correlation -1 between var1 and var2"""
    data = {"var1": [1, 3, 5, 7, 9], "var2": [9, 7, 5, 3, 1]}
    df = pd.DataFrame(data)
    return df


@pytest.fixture
def df_corr_neg1EXPECTED():
    columns = ["var1", "var2"]
    data = [[1.0, -1.0], [-1.0, 1.0]]
    return pd.DataFrame(data, index=columns, columns=columns)


@pytest.fixture
def df_corr_1():
    """Create a synthetic dataset with correlation 1 between var1 and var2"""
    data = {"var1": [1, 2, 4, 8, 16], "var2": [1, 2, 4, 8, 16]}
    df = pd.DataFrame(data)
    return df


@pytest.fixture
def df_corr_1EXPECTED():
    columns = ["var1", "var2"]
    data = [[1.0, 1.0], [1.0, 1.0]]
    return pd.DataFrame(data, index=columns, columns=columns)


@pytest.fixture
def series_corr_neg1():
    """Create a synthetic dataset with correlation -1 between var1 and var2"""
    data = {"var1": [1, 3, 5, 7, 9], "var2": [9, 7, 5, 3, 1]}
    df = pd.DataFrame(data)
    return df["var1"], df["var2"]


@pytest.fixture
def series_corr_1():
    """Create a synthetic dataset with correlation 1 between var1 and var2"""
    data = {"var1": [1, 2, 4, 8, 16], "var2": [1, 2, 4, 8, 16]}
    df = pd.DataFrame(data)
    return df["var1"], df["var2"]


@pytest.fixture
def df_5_columns_corr_100pct_50pct_0pct_neg50pct():
    """Create a synthetic dataset with correlation 1 between var1 and var2, 1/2 between var1 and var3, 0 between var1 and var4, and -1 between var1 and var5"""
    data = {
        "var1": [1, 2, 4, 8, 16],
        "var2": [1, 2, 4, 8, 16],
        "var3": [1, 1.5, 2, 2.5, 3],
        "var4": [10, 1.04, -5, -3, 15],
        "var5": [16, 8, 4, 2, 1],
    }
    df = pd.DataFrame(data)
    return df


@pytest.fixture
def df_5_columns_corr_100pct_50pct_0pct_neg50pctEXPECTED():
    data = {
        "var1": [1, 2, 4, 8, 16],
        "var2": [1, 2, 4, 8, 16],
        "var3": [1, 1.5, 2, 2.5, 3],
        "var4": [10, 1.04, -5, -3, 15],
        "var5": [16, 8, 4, 2, 1],
    }
    df = pd.DataFrame(data)
    expected_corr = np.ones((5, 5))
    for i, j in itertools.combinations(range(5), 2):
        expected_corr[i, j] = pearsonr(df.iloc[:, i], df.iloc[:, j])[0]
        expected_corr[j, i] = expected_corr[i, j]

    return pd.DataFrame(expected_corr, index=df.columns, columns=df.columns)


@pytest.mark.parametrize(
    "df_name, expected",
    [
        ("df_corr_neg1", "df_corr_neg1EXPECTED"),
        ("df_corr_1", "df_corr_1EXPECTED"),
        (
            "df_5_columns_corr_100pct_50pct_0pct_neg50pct",
            "df_5_columns_corr_100pct_50pct_0pct_neg50pctEXPECTED",
        ),
    ],
)
def test_calc_continuous_continuous_corr_df(request, df_name, expected):
    """Test calc_continuous_continuous_corr_df"""
    df = request.getfixturevalue(df_name)
    expected = request.getfixturevalue(expected)
    print(df)
    actual = calc_continuous_continuous_corr(df)
    assert_frame_equal(actual, expected), f"Expected:\n{expected}\nActual:\n{actual}"


@pytest.mark.parametrize(
    "series_name, expected",
    [
        ("series_corr_neg1", -1.0),
        ("series_corr_1", 1.0),
    ],
)
def test_calc_continuous_continuous_corr_series(request, series_name, expected):
    """Test calc_continuous_continuous_corr_series"""
    series1, series2 = request.getfixturevalue(series_name)
    actual = np.round(calc_continuous_continuous_corr(series1, series2), 5)
    assert actual == expected, f"Expected:\n{expected}\nActual:\n{actual}"
