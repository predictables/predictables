import pytest
import os
from predictables.fit_catboost_time_series import (
    get_cat_col_from_filename,
    start_idx_generator,
    idx_to_column_name,
    N_COLS,
)


def test_get_cat_col_from_filename():
    """Test that the function correctly extracts the categorical column name from the filename."""
    # Arrange
    filename = "first_insured_state_id_18_lags.parquet"
    expected = "first_insured_state_id"
    # Act
    result = get_cat_col_from_filename(filename)
    # Assert
    assert result == expected, f"Expected {expected}, but got {result}."


def test_get_cat_col_from_filename_no_lags():
    """Test that the function returns the entire filename if there are no lags."""
    # Arrange
    filename = "first_insured_state_id.parquet"
    expected = "first_insured_state_id"
    # Act
    result = get_cat_col_from_filename(filename)
    # Assert
    assert result == expected, f"Expected {expected}, but got {result}."


def test_get_cat_col_from_filename_no_ncols():
    """Test that the function returns the entire filename if N_COLS is not in the filename."""
    # Arrange
    filename = "first_insured_state_id_lags.parquet"
    expected = "first_insured_state_id_lags"
    # Act
    result = get_cat_col_from_filename(filename)
    # Assert
    assert result == expected, f"Expected {expected}, but got {result}."


@pytest.mark.parametrize(
    "prior_p, expected",
    [
        (6, list(range(1, N_COLS - 6 + 1))),
        (1, list(range(1, N_COLS))),
        (0, list(range(1, N_COLS + 1))),
    ],
)
def test_start_idx_generator(prior_p, expected):
    """Test the start_idx_generator function with different inputs."""
    result = list(start_idx_generator(prior_p))
    assert result == expected, f"Expected {expected}, but got {result}."


@pytest.mark.parametrize(
    "idx, file, expected",
    [
        (
            6,
            "first_insured_state_id_18_lags.parquet",
            "logit[MEAN_ENCODED_first_insured_state_id_360]",
        ),
        (
            1,
            "second_insured_state_id_10_lags.parquet",
            "logit[MEAN_ENCODED_second_insured_state_id_270]",
        ),
        (
            0,
            "third_insured_state_id_5_lags.parquet",
            "logit[MEAN_ENCODED_third_insured_state_id_150]",
        ),
    ],
)
def test_idx_to_column_name(idx, file, expected):
    """Test the idx_to_column_name function with different inputs."""
    result = idx_to_column_name(idx, file)
    assert result == expected, f"Expected {expected}, but got {result}."