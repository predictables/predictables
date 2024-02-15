import pandas as pd
import polars as pl
import pytest
from pandas.testing import assert_frame_equal as pd_assert_frame_equal
from polars.testing import assert_frame_equal as pl_assert_frame_equal

from predictables.impute.src._impute_with_median import impute_with_median

###################################### impute_with_median.py #########################################################################################################


@pytest.fixture
def pd_df():
    return pd.DataFrame({"col": pd.Series([1, 2, 3, 4, 5, None])})


@pytest.fixture
def pl_df(pd_df):
    return pl.from_pandas(pd_df)


@pytest.fixture
def pd_string_df():
    return pd.DataFrame({"col": pd.Series(["a", "b", "c", "d", "a", None])})


@pytest.fixture
def pl_string_df(pd_string_df):
    return pl.from_pandas(pd_string_df)


@pytest.fixture
def pd_numeric_df():
    return pd.DataFrame({"col": pd.Series([1, 2, 3, 4, 5, None])})


@pytest.fixture
def pl_numeric_df(pd_numeric_df):
    return pl.from_pandas(pd_numeric_df)


@pytest.fixture
def pd_non_numeric_df():
    return pd.DataFrame({"col": pd.Series(["a", "b", "c", "d", "a", None])})


@pytest.fixture
def pl_non_numeric_df(pd_non_numeric_df):
    return pl.from_pandas(pd_non_numeric_df)


@pytest.fixture
def pd_null_only_df():
    return pd.DataFrame({"col": pd.Series([None, None, None, None])})


@pytest.fixture
def pl_null_only_df(pd_null_only_df):
    return pl.from_pandas(pd_null_only_df)


def test_impute_with_median_non_numeric_pd_df(pd_non_numeric_df):
    # If a column is non-numeric, the column is returned unchanged
    result = impute_with_median(pd_non_numeric_df)
    assert isinstance(
        result, pl.LazyFrame
    ), f"to_pl_s({type(pd_non_numeric_df)}) is not a pl.LazyFrame"
    pd_assert_frame_equal(result.collect().to_pandas(), pd_non_numeric_df)


def test_impute_with_median_non_numeric_pl_df(pl_non_numeric_df):
    # If a column is non-numeric, the column is returned unchanged
    result = impute_with_median(pl_non_numeric_df)
    assert isinstance(
        result, pl.LazyFrame
    ), f"to_pl_s({type(pl_non_numeric_df)}) is not a pl.LazyFrame"
    pl_assert_frame_equal(result.collect(), pl_non_numeric_df)


def test_impute_with_median_null_only_pd_df(pd_null_only_df):
    result = impute_with_median(pd_null_only_df)
    assert (
        result.collect()["col"].null_count() == pd_null_only_df.shape[0]
    ), "All values should remain null after imputing a null-only DataFrame."
    pd_assert_frame_equal(
        result.collect().to_pandas(),
        pd.DataFrame({"col": pd.Series([None] * 4).astype("object")}),
    )


def test_impute_with_median_null_only_pl_df(pl_null_only_df):
    result = impute_with_median(pl_null_only_df)
    assert (
        result.collect()["col"].null_count() == result.collect().shape[0]
    ), "All values should remain null after imputing a null-only DataFrame."
    pl_assert_frame_equal(
        result.collect(),
        pl.DataFrame({"col": pl.Series([None] * 4).cast(pl.Utf8)}),
    )


def test_impute_with_median_bimodal_pd_df():
    bimodal_df = pd.DataFrame({"col": pd.Series([1, 1, 2, 2, None])})
    # Assuming the bimodal_df is converted inside the function to pl.Series
    result = impute_with_median(bimodal_df)
    median_value = pd.Series([1, 1, 2, 2]).median()
    assert (
        result.collect()["col"][4] == median_value
    ), f"The imputed value ({result.collect()['col'][4]}) should be one of the median ({median_value})."
    pd_assert_frame_equal(
        result.collect().to_pandas(),
        pd.DataFrame({"col": pd.Series([1, 1, 2, 2, median_value])}),
    )


def test_impute_with_median_bimodal_pl_df():
    bimodal_df = pd.DataFrame({"col": pd.Series([1, 1, 2, 2, None])})
    result = impute_with_median(bimodal_df)
    median_value = (
        result.with_columns(pl.col("col").median().alias("median"))
        .select("median")
        .collect()
        .get_column("median")[0]
    )
    assert (
        median_value == 1.5
    ), f"The imputed value ({median_value}) should be the median of [1, 1, 2, 2] (1.5)."
    pl_assert_frame_equal(
        result.collect(), pl.DataFrame({"col": pl.Series([1, 1, 2, 2, 1.5])})
    )


def test_impute_with_median_with_numeric_pd_df(pd_df):
    result = impute_with_median(pd_df)
    assert isinstance(
        result, pl.LazyFrame
    ), f"to_pl_s({type(pd_df)}) is not a pl.LazyFrame"
    assert result.collect().shape[0] == len(
        pd_df
    ), f"Length of result ({result.collect().shape[0]}) does not match length of input ({len(pd_df)})."
    assert (
        result.collect().shape == pd_df.shape
    ), f"Resulting shape ({result.collect().shape}) does not match expected shape ({pd_df.shape})."
    assert (
        result.collect()["col"].dtype == pl.from_pandas(pd_df["col"]).dtype
    ), f"Resulting dtype ({result.collect()['col'].dtype}) does not match expected dtype ({pl.from_pandas(pd_df['col']).dtype})."
    assert (
        result.collect()["col"][5] == 3
    ), f"Imputed value ({result.collect()['col'][5]}) does not match expected value (3)."


def test_impute_with_median_with_numeric_pl_df(pl_df):
    result = impute_with_median(pl_df)
    assert isinstance(
        result, pl.LazyFrame
    ), f"to_pl_s({type(pl_df)}) is not a pl.LazyFrame"
    assert result.collect().shape[0] == len(
        pl_df
    ), f"Length of result ({result.collect().shape[0]}) does not match length of input ({len(pl_df)})."
    assert (
        result.collect().shape == pl_df.shape
    ), f"Resulting shape ({result.collect().shape}) does not match expected shape ({pl_df.shape})."
    assert (
        result.collect()["col"].dtype == pl_df["col"].dtype
    ), f"Resulting dtype ({result.collect()['col'].dtype}) does not match expected dtype ({pl_df['col'].dtype})."
    assert (
        result.collect()["col"][5] == 3
    ), f"Imputed value ({result.collect()['col'][5]}) does not match expected value (3)."


def test_impute_with_median_with_string_pd_df(pd_string_df):
    # if a col is not numeric, will just be returned as is
    result = impute_with_median(pd_string_df)
    assert isinstance(
        result, pl.LazyFrame
    ), f"to_pl_s({type(pd_string_df)}) is not a pl.LazyFrame"
    pd_assert_frame_equal(result.collect().to_pandas(), pd_string_df)


def test_impute_with_median_with_string_pl_df(pl_string_df):
    # if a col is not numeric, will just be returned as is
    result = impute_with_median(pl_string_df)
    assert isinstance(
        result, pl.LazyFrame
    ), f"to_pl_s({type(pl_string_df)}) is not a pl.LazyFrame"
    pl_assert_frame_equal(result.collect(), pl_string_df)


def test_impute_with_median_with_empty_df():
    s = pd.DataFrame({"col": pd.Series([])})
    result = impute_with_median(s)
    assert isinstance(result, pl.LazyFrame), f"to_pl_s({type(s)}) is not a pl.LazyFrame"
    assert (
        result.collect().shape[0] == 0
    ), f"Length of result ({result.collect().shape[0]}) does not match length of input ({s.collect().shape[0]})."


def test_impute_with_median_with_single_value_df():
    s = pd.DataFrame({"col": pd.Series([1])})
    result = impute_with_median(s)
    assert isinstance(
        result, pl.LazyFrame
    ), f"to_pl_s({type(result)}) is not a pl.LazyFrame"
    assert result.collect().shape[0] == 1
    assert (
        result.collect()["col"][0] == 1
    ), f"Imputed value ({result[0]}) does not match expected value (1)."


def test_impute_with_median_with_invalid_input_type():
    with pytest.raises(TypeError):
        impute_with_median(123)


def test_impute_with_median_empty_dataframe():
    empty_df = pd.DataFrame({"col": []})
    result = impute_with_median(empty_df)
    expected = pl.DataFrame({"col": []}).with_columns(
        pl.col("col").cast(pl.Float64).name.keep()
        if pl.__version__ >= "0.19.12"
        else pl.col("col").cast(pl.Float64).keep_name()
    )
    pl_assert_frame_equal(result.collect(), expected)


def test_impute_with_median_data_already_imputed(pd_df):
    result = impute_with_median(pd_df)
    pd_df["col"] = pd_df["col"].fillna(pd_df["col"].copy().median())
    expected = pl.from_pandas(pd_df)
    pl_assert_frame_equal(result.collect(), expected)


def test_impute_with_median_skewed_distribution():
    skewed_df = pd.DataFrame({"col": [1, 2, 2, 2, 2, None, None, None]})
    result = impute_with_median(skewed_df)
    median_value = 2
    expected = pl.DataFrame(
        {"col": [1, 2, 2, 2, 2, median_value, median_value, median_value]}
    )
    expected = expected.with_columns(
        pl.col("col").cast(pl.Float64).name.keep()
        if pl.__version__ >= "0.19.12"
        else pl.col("col").cast(pl.Float64).keep_name()
    )
    pl_assert_frame_equal(result.collect(), expected)
