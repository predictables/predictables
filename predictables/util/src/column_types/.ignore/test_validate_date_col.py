from __future__ import annotations
import pytest
import pandas as pd
import polars as pl
from datetime import datetime
from predictables.util.src.column_types._validate_date_col import (
    validate_date_col,
    validate_pd,
    validate_pl,
)


# Fixtures for pandas and polars dataframes
@pytest.fixture(params=[pd, pl])
def df_module(request):
    return request.param


@pytest.fixture
def date_df(df_module):
    return df_module.DataFrame(
        {"date": [datetime(2022, 1, 1), datetime(2022, 1, 2)], "not_date": [1, 2]}
    )


@pytest.fixture
def not_date_df(df_module):
    return df_module.DataFrame({"not_date": [1, 2]})


# Test validate_pd and validate_pl functions
@pytest.mark.parametrize("validate_func", [validate_pd, validate_pl])
def test_validate_date_col(date_df, not_date_df, string_date_df, validate_func):
    """Test the validate_pd and validate_pl functions."""
    assert validate_func(date_df) is not None  # Has a date column
    assert validate_func(not_date_df) is not None  # Doesn't have a date column
    with pytest.raises(ValueError):
        validate_func(
            string_date_df
        )  # Has a string date column that can't be coerced to a date


# Test validate_date_col decorator
def test_validate_date_col_decorator(date_df, not_date_df, string_date_df):
    """Test the validate_date_col decorator applied to a function."""

    @validate_date_col
    def dummy_func(df) -> pd.DataFrame | pl.DataFrame:
        return df

    assert dummy_func(date_df) is not None  # Has a date column
    assert dummy_func(not_date_df) is not None  # Doesn't have a date column
    with pytest.raises(ValueError):
        dummy_func(
            string_date_df
        )  # Has a string date column that can't be coerced to a date


def test_validate_date_col_no_df():
    """Test that an error is raised if the function is called without a dataframe at all."""

    @validate_date_col
    def dummy_func() -> pd.DataFrame:
        return pd.DataFrame()

    with pytest.raises(ValueError):
        dummy_func(None)


def test_validate_date_col_misplaced_df():
    """Test that an error is raised if the dataframe is passed as a positional argument."""

    @validate_date_col
    def dummy_func(another_arg: int, df: pd.DataFrame) -> pd.DataFrame:
        df["new_col"] = df["date"].dt.day + another_arg
        return df

    with pytest.raises(ValueError):
        dummy_func(1, pd.DataFrame({"date": [datetime(2022, 1, 1)]}))