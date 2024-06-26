from datetime import datetime

import pandas as pd
import polars as pl
import pytest

from predictables.util.src._get_column_dtype import (
    get_column_dtype,
    is_binary,
    is_categorical,
    is_datetime,
    is_integer,
    is_numeric,
)


@pytest.fixture
def numeric_pd_series():
    """Treat this as float.

    This is numeric, but not integer or categorical.
    """
    return pd.Series([1.3, 2.2, 4.1, 8.4, 16.5])


@pytest.fixture
def integer_pd_series():
    """Treat this as an integer, not categorical.

    This is numeric, integer, but not categorical.
    """
    return pd.Series([1, 2, 4, 8, 16])


@pytest.fixture
def non_numeric_pd_series():
    """Treat this as categorical, since it is not numeric."""
    return pd.Series(["a", "b", "c", "d", "e"])


@pytest.fixture
def numeric_pandas_series():
    """Treat this as float.

    This is numeric, but not integer or categorical.
    """
    return pd.Series([1.3, 2.2, 4.1, 8.4, 16.5])


@pytest.fixture
def integer_pandas_series():
    """Treat this as an integer, not categorical.

    This is numeric, integer, but not categorical because the difference between
    unique values is not 1.
    """
    return pd.Series([1, 2, 4, 8, 16]).astype("int")


@pytest.fixture
def non_numeric_pandas_series():
    """Treat this as categorical, since it is not numeric."""
    return pd.Series(["a", "b", "c", "d", "e"])


@pytest.fixture
def numeric_polars_series():
    """Treat this as float.

    It is numeric, but not integer or categorical.
    """
    return pl.Series("numeric", [1.3, 2.2, 4.1, 8.4, 16.5])


@pytest.fixture
def integer_polars_series():
    """Treat this as an integer, not categorical.

    This is numeric, integer, but not categorical because the difference between
    unique values is not 1.
    """
    return pl.Series("integer", [1, 2, 4, 8, 16]).cast(pl.Int64)


@pytest.fixture
def non_numeric_polars_series():
    """Treat this as categorical, since it is not numeric."""
    return pl.Series("non_numeric", ["a", "b", "c", "d", "e"])


@pytest.fixture
def date_pd_series():
    """Treat this as a date series."""
    return pd.Series(
        [
            datetime(2021, 1, 1),
            datetime(2021, 1, 2),
            datetime(2021, 1, 3),
            datetime(2021, 1, 4),
            datetime(2021, 1, 5),
        ]
    )


@pytest.fixture
def date_pandas_series():
    """Treat this as a date series."""
    return pd.Series(
        [
            datetime(2021, 1, 1),
            datetime(2021, 1, 2),
            datetime(2021, 1, 3),
            datetime(2021, 1, 4),
            datetime(2021, 1, 5),
        ]
    )


@pytest.fixture
def date_polars_series():
    """Treat this as a date series."""
    return pl.Series(
        "date",
        [
            datetime(2021, 1, 1),
            datetime(2021, 1, 2),
            datetime(2021, 1, 3),
            datetime(2021, 1, 4),
            datetime(2021, 1, 5),
        ],
    )


@pytest.fixture
def date_numpy_array():
    """Treat this as a date series."""
    return pd.Series(
        [
            datetime(2021, 1, 1),
            datetime(2021, 1, 2),
            datetime(2021, 1, 3),
            datetime(2021, 1, 4),
            datetime(2021, 1, 5),
        ]
    )


@pytest.fixture
def date_tuple():
    """Treat this as a date series."""
    return pd.Series(
        [
            datetime(2021, 1, 1),
            datetime(2021, 1, 2),
            datetime(2021, 1, 3),
            datetime(2021, 1, 4),
            datetime(2021, 1, 5),
        ]
    )


@pytest.fixture
def date_as_string_pandas_series():
    """Treat a series of dates encoded as strings as a date, not categorical.

    This is a date encoded as a string. Should be able to be parsed as a date.
    """
    return pd.Series(
        ["2021-01-01", "2021-01-02", "2021-01-03", "2021-01-04", "2021-01-05"]
    )


@pytest.fixture
def date_as_string_polars_series():
    """Treat a series of dates encoded as strings as a date, not categorical.

    This is a date encoded as a string. Should be able to be parsed as a date.
    """
    return pl.Series(
        "date", ["2021-01-01", "2021-01-02", "2021-01-03", "2021-01-04", "2021-01-05"]
    )


@pytest.fixture
def date_as_string_numpy_array():
    """Treat a numpy array of dates as a date, not categorical.

    This is a date encoded as a string. Should be able to be parsed as a date.
    """
    return pd.Series(
        ["2021-01-01", "2021-01-02", "2021-01-03", "2021-01-04", "2021-01-05"]
    )


@pytest.fixture
def date_as_categorical_pandas_series():
    """Treating a series of categorical dates as a date, not categorical.

    This is a date encoded as a categorical. Should be able to be parsed as a date.
    """
    return pd.Series(
        ["2021-01-01", "2021-01-02", "2021-01-03", "2021-01-04", "2021-01-05"]
    ).astype("category")


@pytest.fixture
def date_as_categorical_polars_series():
    """Treat a series of categorical dates as a date, not categorical.

    This is a date encoded as a categorical. Should be able to be parsed as a date.
    """
    return pl.Series(
        "date", ["2021-01-01", "2021-01-02", "2021-01-03", "2021-01-04", "2021-01-05"]
    ).cast(pl.Categorical)


@pytest.fixture
def binary_pd_series_ints():
    """Treat a pd.Series of 1's and 0's as binary.

    This is a pd_series of integers, whose maximum difference between unique values is 1, but
    there are only two unique values. This means that it is binary.
    """
    return pd.Series([0, 1, 0, 1, 0])


@pytest.fixture
def binary_pd_series_strings():
    """Treating a pd.Series of 1's and 0's as binary.

    This is a pd_series of strings, whose maximum difference between unique values is 1, but
    there are only two unique values. This means that it is binary.
    """
    return pd.Series(["0", "1", "0", "1", "0"])


@pytest.fixture
def binary_pd_series_categorical():
    """Treat a pd.Series of categorical 1's and 0's as binary.

    This is a pandas series of categoricals, whose maximum difference between unique values is 1, but
    there are only two unique values. This means that it is binary.
    """
    return pd.Series(["0", "1", "0", "1", "0"]).astype("category")


@pytest.fixture
def binary_pd_series_booleans():
    """Treat a pd.Series of booleans as binary.

    This is a pd_series of booleans. This means that it is binary.
    """
    return pd.Series([False, True, False, True, False])


@pytest.fixture
def binary_pd_series_floats():
    """Treat a pd.Series of 1.0 and 0.0 as binary.

    This is a pd_series of floats, whose maximum difference between unique values is 1, but
    there are only two unique values. This means that it is binary.
    """
    return pd.Series([0.0, 1.0, 0.0, 1.0, 0.0])


@pytest.fixture
def binary_pd_series_not_0_or_1():
    """Treat pd.Series of exactly two unique values as binary.

    This is a pd_series of floats, whose maximum difference between unique values is 1, but
    there are only two unique values. This means that it is binary.
    """
    return pd.Series([2, 3, 2, 3, 2])  # only two unique values, so should be binary


@pytest.fixture
def categorical_pandas_series():
    """Treat this as categorical, since it is not numeric."""
    return pd.Series(["a", "b", "c", "d", "e"]).astype("category")


@pytest.fixture
def categorical_polars_series():
    """Treat this as categorical, since it is not numeric."""
    return pl.Series("categorical", ["a", "b", "c", "d", "e"]).cast(pl.Categorical)


@pytest.fixture
def text_pandas_series():
    """Treat this as categorical, since it is not numeric."""
    return pd.Series(["a", "b", "c", "d", "e"])


@pytest.fixture
def text_polars_series():
    """Treat this as categorical, since it is not numeric."""
    return pl.Series("text", ["a", "b", "c", "d", "e"]).cast(pl.Utf8)


@pytest.mark.parametrize(
    "series_name, expected",
    [
        ("numeric_pd_series", True),
        ("integer_pd_series", True),
        ("non_numeric_pd_series", False),
        ("numeric_pandas_series", True),
        ("integer_pandas_series", True),
        ("non_numeric_pandas_series", False),
        ("numeric_polars_series", True),
        ("integer_polars_series", True),
        ("non_numeric_polars_series", False),
        ("date_pd_series", False),
        ("date_pandas_series", False),
        ("date_polars_series", False),
        ("date_numpy_array", False),
        ("date_tuple", False),
        ("date_as_string_pandas_series", False),
        ("date_as_string_polars_series", False),
        ("date_as_string_numpy_array", False),
        ("date_as_categorical_pandas_series", False),
        ("date_as_categorical_polars_series", False),
        ("binary_pd_series_ints", True),
        ("binary_pd_series_strings", True),
        ("binary_pd_series_categorical", True),
        ("binary_pd_series_booleans", False),
        ("binary_pd_series_floats", True),
        ("binary_pd_series_not_0_or_1", True),
        ("categorical_pandas_series", False),
        ("categorical_polars_series", False),
        ("text_pandas_series", False),
        ("text_polars_series", False),
    ],
)
def test_is_numeric(request: pytest.FixtureRequest, series_name: str, expected: bool):
    s = request.getfixturevalue(series_name)

    if expected:
        assert is_numeric(
            s
        ), f"Expected is_numeric([{series_name}]) to return {expected}"
    else:
        assert not is_numeric(
            s
        ), f"Expected is_numeric([{series_name}]) to return {expected}"


# Test for is_integer
@pytest.mark.parametrize(
    "series_name, expected",
    [
        ("numeric_pd_series", False),
        ("integer_pd_series", True),
        ("non_numeric_pd_series", False),
        ("numeric_pandas_series", False),
        ("integer_pandas_series", True),
        ("non_numeric_pandas_series", False),
        ("numeric_polars_series", False),
        ("integer_polars_series", True),
        ("non_numeric_polars_series", False),
        ("date_pd_series", False),
        ("date_pandas_series", False),
        ("date_polars_series", False),
        ("date_numpy_array", False),
        ("date_tuple", False),
        ("date_as_string_pandas_series", False),
        ("date_as_string_polars_series", False),
        ("date_as_string_numpy_array", False),
        ("date_as_categorical_pandas_series", False),
        ("date_as_categorical_polars_series", False),
        ("binary_pd_series_ints", True),
        ("binary_pd_series_strings", True),
        ("binary_pd_series_categorical", True),
        ("binary_pd_series_booleans", True),
        (
            "binary_pd_series_floats",
            True,
        ),  # the floats are actually encoded as ints -- they must be for a binary series
        ("binary_pd_series_not_0_or_1", True),
        ("categorical_pandas_series", False),
        ("categorical_polars_series", False),
        ("text_pandas_series", False),
        ("text_polars_series", False),
    ],
)
def test_is_integer(request: pytest.FixtureRequest, series_name: str, expected: bool):
    s = request.getfixturevalue(series_name)

    if expected:
        assert is_integer(
            s
        ), f"Expected is_integer([{series_name}]) to return {expected}"
    else:
        assert not is_integer(
            s
        ), f"Expected is_integer([{series_name}]) to return {expected}"


# Test for is_binary
@pytest.mark.parametrize(
    "series_name, expected",
    [
        ("numeric_pd_series", False),
        ("integer_pd_series", False),
        ("non_numeric_pd_series", False),
        ("numeric_pandas_series", False),
        ("integer_pandas_series", False),
        ("non_numeric_pandas_series", False),
        ("numeric_polars_series", False),
        ("integer_polars_series", False),
        ("non_numeric_polars_series", False),
        ("date_pd_series", False),
        ("date_pandas_series", False),
        ("date_polars_series", False),
        ("date_numpy_array", False),
        ("date_tuple", False),
        ("date_as_string_pandas_series", False),
        ("date_as_string_polars_series", False),
        ("date_as_string_numpy_array", False),
        ("date_as_categorical_pandas_series", False),
        ("date_as_categorical_polars_series", False),
        ("binary_pd_series_ints", True),
        ("binary_pd_series_strings", True),
        ("binary_pd_series_categorical", True),
        ("binary_pd_series_booleans", True),
        (
            "binary_pd_series_floats",
            True,
        ),  # the floats are actually encoded as ints -- they must be for a binary series
        ("binary_pd_series_not_0_or_1", True),
        ("categorical_pandas_series", False),
        ("categorical_polars_series", False),
        ("text_pandas_series", False),
        ("text_polars_series", False),
    ],
)
def test_is_binary(request: pytest.FixtureRequest, series_name: str, expected: bool):
    s = request.getfixturevalue(series_name)

    if expected:
        assert is_binary(s), f"Expected is_binary([{series_name}]) to return {expected}"
    else:
        assert not is_binary(
            s
        ), f"Expected is_binary([{series_name}]) to return {expected}"


# Test for is_datetime
@pytest.mark.parametrize(
    "series_name, expected",
    [
        ("numeric_pd_series", False),
        ("integer_pd_series", False),
        ("non_numeric_pd_series", False),
        ("numeric_pandas_series", False),
        ("integer_pandas_series", False),
        ("non_numeric_pandas_series", False),
        ("numeric_polars_series", False),
        ("integer_polars_series", False),
        ("non_numeric_polars_series", False),
        ("date_pandas_series", True),
        ("date_polars_series", True),
        ("date_as_string_pandas_series", True),
        ("date_as_string_polars_series", True),
        ("date_as_categorical_pandas_series", True),
        ("date_as_categorical_polars_series", True),
        ("binary_pd_series_ints", False),
        ("binary_pd_series_strings", False),
        ("binary_pd_series_categorical", False),
        ("binary_pd_series_booleans", False),
        (
            "binary_pd_series_floats",
            False,
        ),  # the floats are actually encoded as ints -- they must be for a binary series
        ("binary_pd_series_not_0_or_1", False),
        ("categorical_pandas_series", False),
        ("categorical_polars_series", False),
        ("text_pandas_series", False),
        ("text_polars_series", False),
    ],
)
def test_is_datetime(request: pytest.FixtureRequest, series_name: str, expected: bool):
    s = request.getfixturevalue(series_name)

    if expected:
        assert is_datetime(
            s
        ), f"Expected is_datetime([{series_name}]) to return {expected}"
    else:
        assert not is_datetime(
            s
        ), f"Expected is_datetime([{series_name}]) to return {expected}"


# Test for is_categorical
@pytest.mark.parametrize(
    "series_name, expected",
    [
        ("numeric_pd_series", False),
        ("integer_pd_series", False),
        ("non_numeric_pd_series", True),
        ("numeric_pandas_series", False),
        ("integer_pandas_series", False),
        ("non_numeric_pandas_series", True),
        ("numeric_polars_series", False),
        ("integer_polars_series", False),
        ("non_numeric_polars_series", True),
        ("date_pandas_series", False),
        ("date_polars_series", False),
        ("date_as_string_pandas_series", False),
        ("date_as_string_polars_series", False),
        ("date_as_categorical_pandas_series", False),
        ("date_as_categorical_polars_series", False),
        ("binary_pd_series_ints", False),
        ("binary_pd_series_strings", False),
        ("binary_pd_series_categorical", False),
        ("binary_pd_series_booleans", False),
        (
            "binary_pd_series_floats",
            False,
        ),  # the floats are actually encoded as ints -- they must be for a binary series
        ("binary_pd_series_not_0_or_1", False),
        ("categorical_pandas_series", True),
        ("categorical_polars_series", True),
        ("text_pandas_series", True),
        ("text_polars_series", True),
    ],
)
def test_is_categorical(
    request: pytest.FixtureRequest, series_name: str, expected: bool
):
    s = request.getfixturevalue(series_name)

    if expected:
        assert is_categorical(
            s
        ), f"Expected is_categorical([{series_name}]) to return {expected}"
    else:
        assert not is_categorical(
            s
        ), f"Expected is_categorical([{series_name}]) to return {expected}"


# FINALLY, test the function
@pytest.mark.parametrize(
    "series_name, expected",
    [
        ("numeric_pd_series", "continuous"),
        ("integer_pd_series", "continuous"),
        ("non_numeric_pd_series", "categorical"),
        ("numeric_pandas_series", "continuous"),
        ("integer_pandas_series", "continuous"),
        ("non_numeric_pandas_series", "categorical"),
        ("numeric_polars_series", "continuous"),
        ("integer_polars_series", "continuous"),
        ("non_numeric_polars_series", "categorical"),
        ("date_pandas_series", "datetime"),
        ("date_polars_series", "datetime"),
        ("date_as_string_pandas_series", "datetime"),
        ("date_as_string_polars_series", "datetime"),
        ("date_as_categorical_pandas_series", "datetime"),
        ("date_as_categorical_polars_series", "datetime"),
        ("binary_pd_series_ints", "binary"),
        ("binary_pd_series_strings", "binary"),
        ("binary_pd_series_categorical", "binary"),
        ("binary_pd_series_booleans", "binary"),
        (
            "binary_pd_series_floats",
            "binary",
        ),  # the floats are actually encoded as ints -- they must be for a binary series
        ("binary_pd_series_not_0_or_1", "binary"),
        ("categorical_pandas_series", "categorical"),
        ("categorical_polars_series", "categorical"),
        ("text_pandas_series", "categorical"),
        ("text_polars_series", "categorical"),
    ],
)
def test_get_column_dtype2(
    request: pytest.FixtureRequest, series_name: str, expected: str
):
    s = request.getfixturevalue(series_name)

    assert (
        get_column_dtype(s) == expected
    ), f"Expected get_column_dtype([{series_name}]) to return {expected}"
