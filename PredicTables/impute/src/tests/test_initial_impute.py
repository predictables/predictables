import numpy as np
import pandas as pd
import polars as pl
import pytest
from pandas.testing import assert_frame_equal as pd_assert_frame_equal
from polars.testing import assert_frame_equal as pl_assert_frame_equal
from polars.testing import assert_series_equal as pl_assert_series_equal

from PredicTables.impute import (
    get_cv_folds,
    get_missing_data_mask,
    impute_with_median,
    impute_with_mode,
    initial_impute,
)


@pytest.fixture
def pd_df():
    return pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": [6, 7, 8, 9, 10]})


@pytest.fixture
def bigger_pd_df():
    return pd.DataFrame({"a": np.random.randn(1000), "b": np.random.randn(1000)})


@pytest.fixture
def pl_df(pd_df):
    return pl.from_pandas(pd_df)


@pytest.fixture
def bigger_pl_df(bigger_pd_df):
    return pl.from_pandas(bigger_pd_df)


@pytest.fixture
def pl_lf(pl_df):
    return pl_df.lazy()


@pytest.fixture
def bigger_pl_lf(bigger_pl_df):
    return bigger_pl_df.lazy()


# @pytest.fixture
# def rf_classifier():
#     return RandomForestClassifier(n_estimators=10, max_depth=5)


# @pytest.fixture
# def rf_regressor():
#     return RandomForestRegressor(n_estimators=10, max_depth=5)


def test_get_cv_folds_with_pandas_dataframe(pd_df):
    result = get_cv_folds(pd_df, n_folds=2)
    n_rows = pd_df.shape[0]
    assert (
        len(result) == n_rows
    ), f"Length of result ({len(result)}) does not match length of input ({n_rows})."
    assert (
        result.dtype == pl.Int64
    ), f"Resulting dtype ({result.dtype}) does not match expected dtype (pl.Int64)."
    assert (
        result.unique().shape[0] == 2
    ), f"Number of unique values ({result.unique().shape[0]}) does not match expected number of unique values (2)."


def test_get_cv_folds_with_polars_dataframe(pl_df):
    result = get_cv_folds(pl_df, n_folds=2)
    n_rows = pl_df.shape[0]
    assert (
        len(result) == n_rows
    ), f"Length of result ({len(result)}) does not match length of input ({n_rows})."
    assert (
        result.dtype == pl.Int64
    ), f"Resulting dtype ({result.dtype}) does not match expected dtype (pl.Int64)."
    assert (
        (result.unique().shape[0] >= 1) & (result.unique().shape[0] <= 2)
    ), f"Number of unique values ({result.unique().shape[0]}) does not match expected number of unique values (2)."


def test_get_cv_folds_with_polars_lazyframe(pl_lf):
    result = get_cv_folds(pl_lf, n_folds=2)
    n_rows = pl_lf.collect().shape[0]
    assert (
        len(result) == n_rows
    ), f"Length of result ({len(result)}) does not match length of input ({n_rows})."
    assert (
        result.dtype == pl.Int64
    ), f"Resulting dtype ({result.dtype}) does not match expected dtype (pl.Int64)."
    assert (
        result.unique().shape[0] == 2
    ), f"Number of unique values ({result.unique().shape[0]}) does not match expected number of unique values (2)."


def test_get_cv_folds_with_empty_dataframe():
    df = pd.DataFrame()
    result = get_cv_folds(df)
    assert (
        len(result) == 0
    ), f"Length of result ({len(result)}) does not match length of input ({len(df)})."


def test_get_cv_folds_with_single_row_dataframe():
    df = pd.DataFrame({"a": [1], "b": [2]})
    result = get_cv_folds(df)
    assert (
        len(result) == 1
    ), f"Length of result ({len(result)}) does not match length of input ({len(df)})."


def test_get_cv_folds_with_polardf_of_single_row():
    df = pl.DataFrame({"a": [1], "b": [2]})
    result = get_cv_folds(df)
    assert (
        len(result) == 1
    ), f"Length of result ({len(result)}) does not match length of input ({len(df)})."


def test_get_cv_folds_with_polardf_of_empty_dataframe():
    df = pl.DataFrame()
    result = get_cv_folds(df)
    assert (
        len(result) == 0
    ), f"Length of result ({len(result)}) does not match length of input ({len(df)})."


def test_get_cv_folds_with_polardf_of_single_column(bigger_pd_df):
    df = bigger_pd_df[["a"]]
    result = get_cv_folds(df)
    dedup = result.unique().sort().to_numpy()
    assert (
        len(result) == bigger_pd_df.shape[0]
    ), f"Length of result ({len(result)}) does not match length of input ({bigger_pd_df.shape[0]})."
    assert (
        dedup[0] == 1
    ), f"Fold number ({dedup[0]}) does not match expected fold number (1)."
    assert (
        dedup[1] == 2
    ), f"Fold number ({dedup[1]}) does not match expected fold number (3)."
    assert (
        dedup[2] == 3
    ), f"Fold number ({dedup[2]}) does not match expected fold number (2)."
    assert (
        dedup[3] == 4
    ), f"Fold number ({dedup[3]}) does not match expected fold number (4)."
    assert (
        dedup[4] == 5
    ), f"Fold number ({dedup[4]}) does not match expected fold number (5)."


def test_get_cv_folds_with_polardf_of_single_column_and_single_row():
    df = pl.DataFrame({"a": [1]})
    result = get_cv_folds(df)
    assert (
        len(result) == 1
    ), f"Length of result ({len(result)}) does not match length of input ({len(df)})."
    assert (
        result[0] >= 1
    ), f"Fold number ({result[0]}) is not larger than the lowest possible fold number (1)."


def test_get_cv_folds_with_n_folds_equal_to_one(pd_df):
    result = get_cv_folds(pd_df, n_folds=1)
    assert (
        len(result) == len(pd_df)
    ), f"Length of result ({len(result)}) does not match length of input ({len(pd_df)})."
    assert (
        result.dtype == pl.Int64
    ), f"Resulting dtype ({result.dtype}) does not match expected dtype (pl.Int64)."
    assert (
        result.unique().shape[0] == 1
    ), f"Number of unique values ({result.unique().shape[0]}) does not match expected number of unique values (1)."
    assert (
        result[0] == 1
    ), f"Fold number ({result[0]}) does not match expected fold number (1)."


def test_get_cv_folds_with_n_folds_greater_than_number_of_rows(bigger_pd_df):
    result = get_cv_folds(bigger_pd_df, n_folds=10)
    assert (
        len(result) == len(bigger_pd_df)
    ), f"Length of result ({len(result)}) does not match length of input ({len(bigger_pd_df)})."
    assert (
        result.dtype == pl.Int64
    ), f"Resulting dtype ({result.dtype}) does not match expected dtype (pl.Int64)."
    assert (
        result.unique().sort().to_numpy().shape[0] == 10
    ), f"Number of unique values ({result.unique().sort().to_numpy().shape[0]}) does not match expected number of unique values (10)."


def test_get_cv_folds_with_n_folds_equal_to_number_of_rows(bigger_pd_df):
    result = get_cv_folds(bigger_pd_df, n_folds=5)
    assert (
        len(result) == len(bigger_pd_df)
    ), f"Length of result ({len(result)}) does not match length of input ({len(bigger_pd_df)})."
    assert (
        result.dtype == pl.Int64
    ), f"Resulting dtype ({result.dtype}) does not match expected dtype (pl.Int64)."
    assert (
        result.unique().sort().to_numpy().shape[0] == 5
    ), f"Number of unique values ({result.unique().sort().to_numpy().shape[0] == 5}) does not match expected number of unique values (5)."


def test_get_cv_folds_with_n_folds_less_than_one(pd_df):
    with pytest.raises(ValueError):
        get_cv_folds(pd_df, n_folds=0)


def test_get_cv_folds_with_n_folds_not_an_integer(pd_df):
    with pytest.raises(TypeError):
        get_cv_folds(pd_df, n_folds=2.5)


def test_get_cv_folds_with_invalid_data_type():
    with pytest.raises(TypeError):
        get_cv_folds("not a dataframe")


def test_get_cv_folds_with_null_values_in_dataframe(pd_df):
    pd_df_with_nulls = pd_df.copy()
    pd_df_with_nulls.loc[0, "a"] = None
    result = get_cv_folds(pd_df_with_nulls, n_folds=2).sort().unique().to_numpy()
    expected = np.array([1, 2])
    assert np.array_equal(
        result, expected
    ), f"Fold numbers ({result}) do not match expected fold numbers ({expected})."


def test_get_cv_folds_with_index_not_integers(pd_df):
    pd_df_with_str_index = pd_df.copy()
    pd_df_with_str_index.index = ["row1", "row2", "row3", "row4", "row5", "row6"]
    result = get_cv_folds(pd_df_with_str_index, n_folds=2)
    assert (
        len(result) == len(pd_df_with_str_index)
    ), f"Length of result ({len(result)}) does not match length of input ({len(pd_df_with_str_index)})."


def test_get_cv_folds_with_multiple_columns_and_n_folds(bigger_pd_df):
    bigger_pd_df["c"] = np.random.randn(1000)
    result = get_cv_folds(bigger_pd_df, n_folds=5)
    assert (
        len(result) == len(bigger_pd_df)
    ), f"Length of result ({len(result)}) does not match length of input ({len(bigger_pd_df)})."
    assert (
        result.unique().sort().to_numpy().shape[0] == 5
    ), f"Number of unique values ({result.unique().sort().to_numpy().shape[0]}) does not match expected number of unique values (5)."


###################################### get_missing_mask.py #########################################################################################################


def test_get_missing_data_mask_with_pandas_dataframe(pd_df):
    result = get_missing_data_mask(pd_df)
    rows, cols = pd_df.shape
    assert (
        result.collect().shape[0] == pd_df.shape[0]
    ), f"Number of rows in the missing data mask ({result.collect().shape[0]}) does not match number of rows in the input dataframe ({pd_df.shape[0]})."
    assert (
        result.collect().shape[1] == pd_df.shape[1]
    ), f"Number of columns in the missing data mask ({result.collect().shape[1]}) does not match number of columns in the input dataframe ({pd_df.shape[1]})."
    assert (
        result.dtypes == [pl.Boolean] * cols
    ), f"Resulting dtype ({result.dtypes}) does not match expected dtype ({[pl.Boolean] * cols})."


def test_get_missing_data_mask_with_polars_dataframe(pl_df):
    result = get_missing_data_mask(pl_df)
    rows, cols = pl_df.shape
    assert (
        result.collect().shape[0] == pl_df.shape[0]
    ), f"Number of rows in the missing data mask ({result.collect().shape[0]}) does not match number of rows in the input dataframe ({pl_df.shape[0]})."
    assert (
        result.collect().shape[1] == pl_df.shape[1]
    ), f"Number of columns in the missing data mask ({result.collect().shape[1]}) does not match number of columns in the input dataframe ({pl_df.shape[1]})."
    assert (
        result.dtypes == [pl.Boolean] * cols
    ), f"Resulting dtype ({result.dtypes}) does not match expected dtype ({[pl.Boolean] * cols})."


def test_get_missing_data_mask_with_polars_lazyframe(pl_lf):
    result = get_missing_data_mask(pl_lf)
    rows, cols = pl_lf.collect().shape
    assert (
        result.collect().shape[0] == pl_lf.collect().shape[0]
    ), f"Number of rows in the missing data mask ({result.collect().shape[0]}) does not match number of rows in the input dataframe ({pl_lf.collect().shape[0]})."
    assert (
        result.collect().shape[1] == pl_lf.collect().shape[1]
    ), f"Number of columns in the missing data mask ({result.collect().shape[1]}) does not match number of columns in the input dataframe ({pl_lf.collect().shape[1]})."
    assert (
        result.dtypes == [pl.Boolean] * cols
    ), f"Resulting dtype ({result.dtypes}) does not match expected dtype ({[pl.Boolean] * cols})."


def test_get_missing_data_mask_with_empty_dataframe():
    df = pd.DataFrame()
    result = get_missing_data_mask(df)
    assert (
        result.collect().shape[0] == 0
    ), f"Length of result ({result.collect().shape[0]}) does not match length of input ({0})."
    assert (
        result.collect().shape[1] == 0
    ), f"Length of result ({result.collect().shape[1]}) does not match length of input ({0})."


def test_get_missing_data_mask_with_single_row_dataframe():
    df = pd.DataFrame({"a": [1], "b": [2]})
    rows, cols = df.shape
    result = get_missing_data_mask(df)
    assert (
        result.collect().shape[0] == rows
    ), f"Number of rows in the missing data mask ({result.collect().shape[0]}) does not match number of rows in the input dataframe ({rows})."
    assert (
        result.collect().shape[1] == cols
    ), f"Number of columns in the missing data mask ({result.collect().shape[1]}) does not match number of columns in the input dataframe ({cols})."
    assert (
        result.dtypes == [pl.Boolean] * cols
    ), f"Resulting dtype ({result.dtypes}) does not match expected dtype ({[pl.Boolean] * cols})."
    pd_assert_frame_equal(
        result.collect().to_pandas(), pd.DataFrame({"a": [False], "b": [False]})
    )


def test_get_missing_data_mask_with_polardf_of_single_row():
    df = pl.DataFrame({"a": [1], "b": [2]})
    rows, cols = df.shape
    result = get_missing_data_mask(df)
    assert (
        result.collect().shape[0] == rows
    ), f"Number of rows in the missing data mask ({result.collect().shape[0]}) does not match number of rows in the input dataframe ({rows})."
    assert (
        result.collect().shape[1] == cols
    ), f"Number of columns in the missing data mask ({result.collect().shape[1]}) does not match number of columns in the input dataframe ({cols})."
    assert (
        result.dtypes == [pl.Boolean] * cols
    ), f"Resulting dtype ({result.dtypes}) does not match expected dtype ({[pl.Boolean] * cols})."
    pd_assert_frame_equal(
        result.collect().to_pandas(), pd.DataFrame({"a": [False], "b": [False]})
    )


def test_get_missing_data_mask_with_polardf_of_single_column():
    df = pl.DataFrame({"a": [1, 2, 3, 4, 5]})
    rows, cols = df.shape
    result = get_missing_data_mask(df)
    assert (
        result.collect().shape[0] == rows
    ), f"Number of rows in the missing data mask ({result.collect().shape[0]}) does not match number of rows in the input dataframe ({rows})."
    assert (
        result.collect().shape[1] == cols
    ), f"Number of columns in the missing data mask ({result.collect().shape[1]}) does not match number of columns in the input dataframe ({cols})."
    assert (
        result.dtypes == [pl.Boolean] * cols
    ), f"Resulting dtype ({result.dtypes}) does not match expected dtype ({[pl.Boolean] * cols})."
    pd_assert_frame_equal(
        result.collect().to_pandas(),
        pd.DataFrame({"a": [False, False, False, False, False]}),
    )


def test_get_missing_data_mask_with_polardf_of_single_column_and_single_row():
    df = pl.DataFrame({"a": [1]})
    result = get_missing_data_mask(df)
    assert (
        result.collect().shape[0] == 1
    ), f"Length of result ({result.collect().shape[0]}) does not match length of input ({1})."


def test_get_missing_data_mask_with_bigger_pandas_dataframe(bigger_pd_df):
    result = get_missing_data_mask(bigger_pd_df)
    rows, cols = bigger_pd_df.shape
    assert (
        result.collect().shape[0] == rows
    ), f"Number of rows in result ({result.collect().shape[0]}) does not have the expected number of rows ({rows})."
    assert (
        result.collect().shape[1] == cols
    ), f"Number of columns in result ({result.collect().shape[1]}) does not have the expected number of columns ({cols})."
    assert (
        result.dtypes == [pl.Boolean] * cols
    ), f"Resulting dtype ({result.dtypes}) does not match expected dtype ({[pl.Boolean] * cols})."
    pd_assert_frame_equal(
        result.collect().to_pandas(),
        pd.DataFrame({"a": [False] * 1000, "b": [False] * 1000}),
    )


def test_get_missing_data_mask_with_bigger_polars_dataframe(bigger_pl_df):
    result = get_missing_data_mask(bigger_pl_df)
    rows, cols = bigger_pl_df.shape
    assert (
        result.collect().shape[0] == rows
    ), f"Number of rows in result ({result.collect().shape[0]}) does not have the expected number of rows ({rows})."
    assert (
        result.collect().shape[1] == cols
    ), f"Number of columns in result ({result.collect().shape[1]}) does not have the expected number of columns ({cols})."
    assert (
        result.dtypes == [pl.Boolean] * cols
    ), f"Resulting dtype ({result.dtypes}) does not match expected dtype ({[pl.Boolean] * cols})."


def test_get_missing_data_mask_with_bigger_polars_lazyframe(bigger_pl_lf):
    result = get_missing_data_mask(bigger_pl_lf)
    rows, cols = bigger_pl_lf.collect().shape
    assert (
        result.collect().shape[0] == rows
    ), f"Number of rows in result ({result.collect().shape[0]}) does not have the expected number of rows ({rows})."
    assert (
        result.collect().shape[1] == cols
    ), f"Number of columns in result ({result.collect().shape[1]}) does not have the expected number of columns ({cols})."
    assert (
        result.dtypes == [pl.Boolean] * cols
    ), f"Resulting dtype ({result.dtypes}) does not match expected dtype ({[pl.Boolean] * cols})."


def test_get_missing_data_mask_with_all_values_missing(pd_df):
    pd_df_all_missing = pd_df.copy()
    pd_df_all_missing.loc[:, :] = np.nan
    result = get_missing_data_mask(pd_df_all_missing)
    expected = pd.DataFrame(
        True, index=pd_df_all_missing.index, columns=pd_df_all_missing.columns
    )
    pd_assert_frame_equal(result.collect().to_pandas(), expected)


def test_get_missing_data_mask_with_no_missing_values(bigger_pd_df):
    result = get_missing_data_mask(bigger_pd_df)
    expected = pd.DataFrame(
        False, index=bigger_pd_df.index, columns=bigger_pd_df.columns
    )
    pd_assert_frame_equal(result.collect().to_pandas(), expected)


def test_get_missing_data_mask_with_some_missing_values(pd_df):
    pd_df_some_missing = pd_df.copy()
    pd_df_some_missing.loc[0, "a"] = np.nan
    result = get_missing_data_mask(pd_df_some_missing)
    expected = pd_df_some_missing.isna()
    pd_assert_frame_equal(result.collect().to_pandas(), expected)


def test_get_missing_data_mask_with_non_numeric_columns():
    df_with_strings = pd.DataFrame({"a": ["foo", "bar", None, "baz", "qux"]})
    result = get_missing_data_mask(df_with_strings)
    expected = df_with_strings.isna()
    pd_assert_frame_equal(result.collect().to_pandas(), expected)


def test_get_missing_data_mask_with_inconsistent_index(pd_df):
    pd_df_inconsistent_index = pd_df.copy()
    pd_df_inconsistent_index.index = [10, 11, 12, 13, 14, 15]
    result = get_missing_data_mask(pd_df_inconsistent_index)
    assert (
        result.collect().shape[0] == pd_df_inconsistent_index.shape[0]
    ), f"Number of rows in result ({result.collect().shape[0]}) does not have the expected number of rows ({pd_df_inconsistent_index.shape[0]})."


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
        result.collect(), pl.DataFrame({"col": pl.Series([None] * 4).cast(pl.Utf8)})
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
    assert (
        result.collect().shape[0] == len(pd_df)
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
    assert (
        result.collect().shape[0] == len(pl_df)
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
        pl.col("col").cast(pl.Float64).keep_name()
    )
    pl_assert_frame_equal(result.collect(), expected)


def test_impute_with_median_data_already_imputed(pd_df):
    result = impute_with_median(pd_df)
    pd_df["col"].fillna(pd_df["col"].median(), inplace=True)
    expected = pl.from_pandas(pd_df)
    pl_assert_frame_equal(result.collect(), expected)


def test_impute_with_median_skewed_distribution():
    skewed_df = pd.DataFrame({"col": [1, 2, 2, 2, 2, None, None, None]})
    result = impute_with_median(skewed_df)
    median_value = 2
    expected = pl.DataFrame(
        {"col": [1, 2, 2, 2, 2, median_value, median_value, median_value]}
    )
    expected = expected.with_columns(pl.col("col").cast(pl.Float64).keep_name())
    pl_assert_frame_equal(result.collect(), expected)


###################################### impute_with_mode.py #####################################################################


@pytest.fixture
def pd_categorical_series():
    s = pd.Series(["red", "blue", "blue", "blue", "red", None])
    return s.astype("category")


@pytest.fixture
def pl_categorical_series(pd_categorical_series):
    return pl.from_pandas(pd_categorical_series)


def test_check_if_categorical(pd_categorical_series, pl_categorical_series):
    pl_df = pl.DataFrame({"variable": pl_categorical_series})
    assert _check_if_categorical(
        pl_df, "variable"
    ), "Categorical polars series not recognized as categorical."
    assert _check_if_categorical(
        pd.DataFrame({"variable": pd_categorical_series}), "variable"
    ), "Categorical pandas series not recognized as categorical."


def test_impute_col_with_mode_with_pd_series(pd_categorical_series):
    df = pd.DataFrame({"colors": pd_categorical_series.astype(str)})
    df["colors"].replace({"nan": None}, inplace=True)
    result = impute_with_mode(df)
    # Convert to pandas DataFrame for easier comparison and assert
    result_df = result.collect().to_pandas()

    # Check that no null values remain
    assert not result_df["colors"].isna().any(), "Null values were not imputed."

    # Determine the mode of the non-null values in the original series
    mode_value = pd_categorical_series.mode().iloc[0]

    # Replace NaN in the expected DataFrame with the mode value
    expected_df = pd.DataFrame(
        {"colors": ["red", "blue", "blue", "blue", "red", mode_value]}
    )
    expected_df["colors"] = expected_df["colors"].astype(str)

    # Compare the result with the expected DataFrame
    pd_assert_frame_equal(result_df, expected_df)


def test_impute_col_with_mode_with_pl_series(pl_categorical_series):
    df = pl.DataFrame({"colors": pl_categorical_series.cast(pl.Utf8)})
    result = impute_with_mode(df)
    assert result.collect()["colors"].null_count() == 0, "Null values were not imputed."
    pl_assert_frame_equal(
        result.collect(),
        pl.DataFrame({"colors": ["red", "blue", "blue", "blue", "red", "blue"]}),
    )


def test_impute_with_mode_empty_series():
    empty_series = pd.Series([], dtype="object")
    pl_empty_series = pl.from_pandas(empty_series).to_frame("variable")
    result = impute_with_mode(pl_empty_series)
    expected = pl.DataFrame({"variable": pl.Series([], dtype=pl.Utf8)})
    pl_assert_frame_equal(result.collect(), expected)


def test_impute_with_mode_single_value_series():
    single_value_series = pd.Series(["single_value"], dtype="object")
    pl_single_value_series = pl.from_pandas(single_value_series).to_frame("column_name")
    result = impute_with_mode(pl_single_value_series)
    assert (
        result.collect()["column_name"][0] == "single_value"
    ), "Imputed value should not alter the original single value."
    pl_assert_frame_equal(
        result.collect(),
        pl.DataFrame({"column_name": pl.Series(["single_value"], dtype=pl.Utf8)}),
    )


def test_impute_with_mode_null_only_series():
    null_only_series = pd.Series([None, None, None], dtype="object")
    pl_null_only_series = pl.from_pandas(null_only_series)
    result = impute_with_mode(pl_null_only_series)
    assert (
        result.collect().to_series().null_count() == 3
    ), "Null-only series imputation should not impute values."
    pl_assert_series_equal(
        result.collect().to_series(), pl.Series([None, None, None], dtype=pl.Utf8)
    )


def test_impute_with_mode_invalid_input_type():
    with pytest.raises(TypeError):
        impute_with_mode(123)


def test_impute_with_mode_empty_dataframe():
    empty_df = pd.DataFrame({"col": []})
    result = impute_with_mode(empty_df)
    expected = pl.DataFrame({"col": np.array([], dtype=np.float64)})
    pl_assert_frame_equal(result.collect(), expected)


def test_impute_with_mode_uniform_distribution():
    uniform_df = pd.DataFrame({"col": ["val"] * 10 + [None]})
    result = impute_with_mode(uniform_df)
    expected = pl.DataFrame({"col": ["val"] * 11})
    pl_assert_frame_equal(result.collect(), expected)


def test_impute_with_mode_multiple_modes():
    multimodal_df = pd.DataFrame({"col": ["red", "blue", "red", "blue", None]})
    result = impute_with_mode(multimodal_df)
    # Assuming the function picks the first mode in case of multiple modes
    expected_mode = "blue"
    expected = pl.DataFrame({"col": ["red", "blue", "red", "blue", expected_mode]})
    pl_assert_frame_equal(result.collect(), expected)


###################################### initial_impute.py #####################################################################


# Test for DataFrame with numeric columns
def test_initial_impute_numeric_pd_df(pd_numeric_df):
    result = initial_impute(pd_numeric_df)
    assert isinstance(
        result, pl.LazyFrame
    ), f"Result {result} is not a pl.LazyFrame: {type(result)}"
    pd_assert_frame_equal(
        result.collect().to_pandas(),
        pd.DataFrame({"col": pd.Series([1, 2, 3, 4, 5, 3]).astype(float)}),
    )


# Test for DataFrame with string columns
def test_initial_impute_string_pd_df(pd_string_df):
    result = initial_impute(pd_string_df)
    assert isinstance(
        result, pl.LazyFrame
    ), f"Result {result} is not a pl.LazyFrame: {type(result)}"
    pd_assert_frame_equal(
        result.collect().to_pandas(),
        pd.DataFrame({"col": pd.Series(["a", "b", "c", "d", "a", "a"])}),
    )


# Test for DataFrame with mixed columns
def test_initial_impute_mixed_pd_df(pd_numeric_df, pd_string_df):
    mixed_df = pd_numeric_df.join(pd_string_df, rsuffix="_str")
    result = initial_impute(mixed_df)

    # Assert the result is a LazyFrame
    assert isinstance(
        result, pl.LazyFrame
    ), f"Result {result} is not a pl.LazyFrame: {type(result)}"

    # Collect the result into a pandas DataFrame
    result_df = result.collect().to_pandas()

    # Check that no null values remain in the numeric column
    assert (
        not result_df["col"].isna().any()
    ), "Null values were not imputed in the numeric column."

    # Check that no null values remain in the string column
    assert (
        not result_df["col_str"].isna().any()
    ), "Null values were not imputed in the string column."

    # Determine the mode of the non-null values in the original string series
    mode_value_str = pd_string_df["col"].mode().iloc[0]

    # Create the expected DataFrame with the imputed values
    expected_df = pd.DataFrame(
        {
            "col": pd.Series([1, 2, 3, 4, 5, 3]).astype(float),
            "col_str": pd.Series(["a", "b", "c", "d", "a", mode_value_str]),
        }
    )

    # Compare the result with the expected DataFrame
    pd_assert_frame_equal(result_df, expected_df)

    # mixed_df = pd_numeric_df.join(pd_string_df, rsuffix='_str')
    # result = initial_impute(mixed_df)
    # assert isinstance(result, pl.LazyFrame), f"Result {result} is not a pl.LazyFrame: {type(result)}"
    # pd_assert_frame_equal(result.collect().to_pandas(), pd.DataFrame({'col':pd.Series([1, 2, 3, 4, 5, 3]).astype(float), 'col_str':pd.Series(['a', 'b', 'c', 'd', 'a', 'a'])}))


# Test for empty DataFrame
def test_initial_impute_empty_df():
    empty_df = pd.DataFrame()
    result = initial_impute(empty_df)
    assert isinstance(
        result, pl.LazyFrame
    ), f"Result {result} is not a pl.LazyFrame: {type(result)}"
    assert result.collect().is_empty(), f"Result {result} should be an empty DataFrame"


# Test for DataFrame with all null columns
def test_initial_impute_null_only_pd_df(pd_null_only_df):
    result = initial_impute(pd_null_only_df)
    assert isinstance(
        result, pl.LazyFrame
    ), f"Result {result} is not a pl.LazyFrame: {type(result)}"
    pd_assert_frame_equal(
        result.collect().to_pandas(),
        pd.DataFrame({"col": pd.Series([None, None, None, None])}),
    )


# Test for DataFrame with a single value
def test_initial_impute_single_value_df():
    single_value_df = pd.DataFrame({"col": [1.0]})
    result = initial_impute(single_value_df)
    assert isinstance(
        result, pl.LazyFrame
    ), f"Result {result} is not a pl.LazyFrame: {type(result)}"
    pd_assert_frame_equal(
        result.collect().to_pandas(), pd.DataFrame({"col": pd.Series([1.0])})
    )


# Test for invalid input type
def test_initial_impute_invalid_input_type():
    with pytest.raises(TypeError):
        initial_impute(123)  # Assuming initial_impute cannot handle int


###################################### get_rf_hyperparameters.py ###############################################################

# class TestGetDefaultParamsCol:

#     def test_valid_numeric_pandas(self):
#         df = pd.DataFrame({'numeric_col': [1, 2, 3]})
#         expected = regression_grid()
#         assert _get_default_params_col(df, 'numeric_col') == expected

#     def test_valid_categorical_pandas(self):
#         df = pd.DataFrame({'categorical_col': pd.Categorical(['A', 'B', 'A'])})
#         expected = classification_grid()
#         assert _get_default_params_col(df, 'categorical_col') == expected

#     def test_valid_numeric_polars(self):
#         df = pl.DataFrame({'numeric_col': [1.0, 2.0, 3.0]})
#         expected = regression_grid()
#         assert _get_default_params_col(df, 'numeric_col') == expected

#     def test_valid_string_polars(self):
#         df = pl.DataFrame({'string_col': ['A', 'B', 'A']})
#         expected = classification_grid()
#         assert _get_default_params_col(df, 'string_col') == expected

#     def test_df_none(self):
#         with pytest.raises(ValueError, match="DataFrame cannot be None."):
#             _get_default_params_col(None, 'any_col')

#     def test_col_none(self):
#         df = pd.DataFrame({'numeric_col': [1, 2, 3]})
#         with pytest.raises(ValueError, match="Column name cannot be None."):
#             _get_default_params_col(df, None)

#     def test_col_not_in_df(self):
#         df = pd.DataFrame({'numeric_col': [1, 2, 3]})
#         with pytest.raises(ValueError, match="Column 'not_a_col' not found in DataFrame."):
#             _get_default_params_col(df, 'not_a_col')

#     def test_unsupported_column_type(self):
#         df = pd.DataFrame({'unsupported_col': pd.TimedeltaIndex([1, 2, 3], unit='d')})
#         with pytest.raises(ValueError, match="Column type must be numeric or string, not timedelta64[ns]."):
#             _get_default_params_col(df, 'unsupported_col')
#         df = pd.DataFrame({'numeric_col': [1, 2, 3]})
#         with pytest.raises(ValueError, match="Column 'not_a_col' not found in DataFrame."):
#             _get_default_params_col(df, 'not_a_col')

#     def test_unsupported_column_type(self):
#         df = pd.DataFrame({'unsupported_col': pd.TimedeltaIndex([1, 2, 3], unit='d')})
#         with pytest.raises(ValueError, match="Column type must be numeric or string, not timedelta64[ns]."):
#             _get_default_params_col(df, 'unsupported_col')
