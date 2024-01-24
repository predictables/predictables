import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from predictables.univariate.src._get_data import (
    _filter_df_for_cv,
    _filter_df_for_cv_test,
    _filter_df_for_cv_train,
    _filter_df_for_train_test,
)


@pytest.fixture
def sample_dataframe():
    # Create a sample dataframe for testing
    df = pd.DataFrame(
        {"fold_col": [1, 2, 3, 3, 4, 5], "data": ["A", "B", "C", "c1", "D", "E"]}
    )
    return df


@pytest.fixture
def sample_val_dataframe():
    # Create a sample validation dataframe for testing
    df_val = pd.DataFrame({"data": ["F", "G", "H", "I", "J"]})
    return df_val


@pytest.fixture
def sample_total_dataframe(df, df_val):
    # Create a sample total dataframe for testing
    df_total = pd.DataFrame(
        {
            "fold_col": [1, 2, 3, 3, 4, 5] + ([-42] * df_val.shape[0]),
            "data": ["A", "B", "C", "c1", "D", "E", "F", "G", "H", "I", "J"],
        }
    )
    return df_total


def test_filter_df_for_cv_train_valid_input(sample_dataframe):
    # Test with valid input
    fold = 3
    fold_col = "fold_col"
    expected_output = pd.DataFrame(
        {"fold_col": [1, 2, 4, 5], "data": ["A", "B", "D", "E"]}
    )

    output = _filter_df_for_cv_train(sample_dataframe, fold, fold_col).reset_index(
        drop=True
    )
    (
        assert_frame_equal(output, expected_output),
        f"Expected {expected_output}, got {output}",
    )


def test_filter_df_for_cv_test_valid_input(sample_dataframe):
    # Test with valid input
    fold = 3
    fold_col = "fold_col"
    expected_output = pd.DataFrame({"fold_col": [3, 3], "data": ["C", "c1"]})

    output = _filter_df_for_cv_test(sample_dataframe, fold, fold_col).reset_index(
        drop=True
    )
    (
        assert_frame_equal(output, expected_output),
        f"Expected {expected_output}, got {output}",
    )


@pytest.mark.parametrize(
    "data, fold_col, values",
    [
        ("train", "1 2 4 5", "A B D E"),
        ("test", "3 3", "C c1"),
        ("all", "1 2 3 3 4 5", "A B C c1 D E"),
    ],
)
def test_filter_df_for_cv_valid_input(sample_dataframe, data, fold_col, values):
    # Test with valid input
    fold = 3
    # fold_col = "fold_col"
    expected_output = pd.DataFrame(
        {"fold_col": [int(x) for x in fold_col.split()], "data": values.split()}
    )

    output = _filter_df_for_cv(sample_dataframe, fold, "fold_col", data).reset_index(
        drop=True
    )
    (
        assert_frame_equal(output, expected_output),
        f"Expected {expected_output}, got {output}",
    )


@pytest.mark.parametrize(
    "fold, fold_col, msg",
    [
        (
            1,
            "invalid_col",
            "No column in the DataFrame called 'invalid_col'",
        ),  # Invalid fold column name
        (
            "invalid_fold",
            "fold_col",
            "invalid_fold is not a named cv fold in the DataFrame.",
        ),  # Invalid fold number
        (
            "invalid_fold",
            "invalid_col",
            "No column in the DataFrame called 'invalid_col'",
        ),  # Invalid fold number and column name
    ],
)
def test_filter_df_for_cv_train_invalid_input(sample_dataframe, fold, fold_col, msg):
    # Test with invalid input
    with pytest.raises(KeyError) as err:
        _filter_df_for_cv_train(sample_dataframe, fold, fold_col)
    assert (
        f'"{msg.strip().lower()}"'.replace("'", '"')
        == str(err.value).strip().lower().replace("'", '"')
    ), f"Expected the error to be {msg} for [{fold}]/[{fold_col}].\n\nInstead got {err.value}"


@pytest.mark.parametrize(
    "fold, fold_col, msg",
    [
        (
            1,
            "invalid_col",
            "No column in the DataFrame called 'invalid_col'",
        ),  # Invalid fold column name
        (
            "invalid_fold",
            "fold_col",
            "invalid_fold is not a named cv fold in the DataFrame.",
        ),  # Invalid fold number
        (
            "invalid_fold",
            "invalid_col",
            "No column in the DataFrame called 'invalid_col'",
        ),  # Invalid fold number and column name
    ],
)
def test_filter_df_for_cv_test_invalid_input(sample_dataframe, fold, fold_col, msg):
    # Test with invalid input
    with pytest.raises(KeyError) as err:
        _filter_df_for_cv_test(sample_dataframe, fold, fold_col)
    assert (
        f'"{msg.strip().lower()}"'.replace("'", '"')
        == str(err.value).strip().lower().replace("'", '"')
    ), f"Expected the error to be {msg} for [{fold}]/[{fold_col}].\n\nInstead got {err.value}"


@pytest.mark.parametrize(
    "fold, fold_col, data, msg",
    [
        (
            1,
            "invalid_col",
            "train",
            "No column in the DataFrame called 'invalid_col'",
        ),  # Invalid fold column name
        (
            1,
            "invalid_col",
            "test",
            "No column in the DataFrame called 'invalid_col'",
        ),  # Invalid fold column name
        (
            1,
            "invalid_col",
            "all",
            "No column in the DataFrame called 'invalid_col'",
        ),  # Invalid fold column name
        (
            "invalid_fold",
            "fold_col",
            "train",
            "invalid_fold is not a named cv fold in the DataFrame.",
        ),  # Invalid fold number
        (
            "invalid_fold",
            "fold_col",
            "test",
            "invalid_fold is not a named cv fold in the DataFrame.",
        ),  # Invalid fold number
        (
            "invalid_fold",
            "fold_col",
            "all",
            "invalid_fold is not a named cv fold in the DataFrame.",
        ),  # Invalid fold number
        (
            "invalid_fold",
            "invalid_col",
            "train",
            "No column in the DataFrame called 'invalid_col'",
        ),  # Invalid fold number and column name
        (
            "invalid_fold",
            "invalid_col",
            "test",
            "No column in the DataFrame called 'invalid_col'",
        ),  # Invalid fold number and column name
        (
            "invalid_fold",
            "invalid_col",
            "all",
            "No column in the DataFrame called 'invalid_col'",
        ),  # Invalid fold number and column name
    ],
)
def test_filter_df_for_cv_invalid_input(sample_dataframe, fold, fold_col, data, msg):
    # Test with invalid input
    with pytest.raises(KeyError) as err:
        _filter_df_for_cv(sample_dataframe, fold, fold_col, data)
    assert (
        f'"{msg.strip().lower()}"'.replace("'", '"')
        == str(err.value).strip().lower().replace("'", '"')
    ), f"Expected the error to be {msg} for [{fold}]/[{fold_col}].\n\nInstead got {err.value}"


def test_filter_df_for_train_test_all(sample_dataframe, sample_val_dataframe):
    # Test with data='all'
    expected_output = pd.DataFrame(
        {
            "fold_col": [1, 2, 3, 3, 4, 5] + ([-42] * sample_val_dataframe.shape[0]),
            "data": ["A", "B", "C", "c1", "D", "E", "F", "G", "H", "I", "J"],
        }
    )
    output = _filter_df_for_train_test(
        sample_dataframe, sample_val_dataframe, data="all"
    ).reset_index(drop=True)
    (
        assert_frame_equal(output, expected_output),
        f"Expected {expected_output}, got {output}",
    )


def test_filter_df_for_train_test_train(sample_dataframe, sample_val_dataframe):
    # Test with data='train'
    expected_output = pd.DataFrame(
        {"fold_col": [1, 2, 3, 3, 4, 5], "data": ["A", "B", "C", "c1", "D", "E"]}
    )
    output = _filter_df_for_train_test(
        sample_dataframe, sample_val_dataframe, data="train"
    )
    (
        assert_frame_equal(output, expected_output),
        f"Expected {expected_output}, got {output}",
    )


def test_filter_df_for_train_test_test(sample_dataframe, sample_val_dataframe):
    # Test with data='test'
    expected_output = pd.DataFrame({"data": ["F", "G", "H", "I", "J"]})
    output = _filter_df_for_train_test(
        sample_dataframe, sample_val_dataframe, data="test"
    )
    (
        assert_frame_equal(output, expected_output),
        f"Expected {expected_output}, got {output}",
    )


def test_filter_df_for_train_test_invalid_data(sample_dataframe, sample_val_dataframe):
    # Test with invalid data value
    with pytest.raises(ValueError) as err:
        _filter_df_for_train_test(
            sample_dataframe, sample_val_dataframe, data="invalid"
        )
    assert (
        '"data" must be one of ["all", "train", "test"]. got "invalid".'
        == str(err.value).strip().lower().replace("'", '"')
    ), f"Expected the error to be \"data\" must be one of ['all', 'train', 'test'].\n\nInstead got {err.value}"
