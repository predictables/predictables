from unittest.mock import patch

import pytest

from predictables.util.src._load_env import load_env


@patch("predictables.util.src._load_env.load_dotenv")
@patch("predictables.util.src._load_env.dotenv_values")
def test_load_env(mock_dotenv_values, mock_load_dotenv):
    # Test when .env file is loaded successfully
    mock_load_dotenv.return_value = True
    mock_dotenv_values.return_value = {"VAR1": "value1", "VAR2": "value2"}

    assert load_env() == {"VAR1": "value1", "VAR2": "value2"}, (
        f"Expected: { {'VAR1': 'value1', 'VAR2': 'value2'} }, "
        f"Actual: { load_env() }"
    )

    # Test when .env file fails to load
    mock_load_dotenv.return_value = False
    assert load_env() == {}, f"Expected: { {} }, " f"Actual: { load_env() }"


@patch("predictables.util.src._load_env.load_dotenv")
@patch("predictables.util.src._load_env.dotenv_values")
def test_load_env_empty_file(mock_dotenv_values, mock_load_dotenv):
    mock_load_dotenv.return_value = True
    mock_dotenv_values.return_value = {}
    assert load_env() == {}, "Expected: {}, Actual: {}".format({}, load_env())


@patch("predictables.util.src._load_env.load_dotenv")
@patch("predictables.util.src._load_env.dotenv_values")
def test_load_env_invalid_format(mock_dotenv_values, mock_load_dotenv):
    mock_load_dotenv.return_value = True
    mock_dotenv_values.return_value = {"INVALID_FORMAT": ""}
    with pytest.raises(ValueError):
        load_env()


@patch("predictables.util.src._load_env.load_dotenv")
@patch("predictables.util.src._load_env.dotenv_values")
def test_load_env_special_characters(mock_dotenv_values, mock_load_dotenv):
    mock_load_dotenv.return_value = True
    mock_dotenv_values.return_value = {"VAR": "val$ue#1"}
    assert load_env() == {"VAR": "val$ue#1"}, "Expected: {}, Actual: {}".format(
        {"VAR": "val$ue#1"}, load_env()
    )


@patch("predictables.util.src._load_env.load_dotenv")
@patch("predictables.util.src._load_env.dotenv_values")
def test_load_env_large_values(mock_dotenv_values, mock_load_dotenv):
    mock_load_dotenv.return_value = True
    large_value = "a" * 10000  # a string with 10,000 characters
    mock_dotenv_values.return_value = {"VAR": large_value}
    assert load_env() == {"VAR": large_value}, "Expected: {}, Actual: {}".format(
        {"VAR": large_value}, load_env()
    )


@patch("predictables.util.src._load_env.load_dotenv")
@patch("predictables.util.src._load_env.dotenv_values")
def test_load_env_large_number_of_variables(mock_dotenv_values, mock_load_dotenv):
    mock_load_dotenv.return_value = True
    large_number_of_variables = {
        f"VAR{i}": f"value{i}" for i in range(10000)
    }  # a dictionary with 10,000 variables
    mock_dotenv_values.return_value = large_number_of_variables
    assert load_env() == large_number_of_variables, "Expected: {}, Actual: {}".format(
        large_number_of_variables, load_env()
    )


@patch("predictables.util.src._load_env.load_dotenv")
@patch("predictables.util.src._load_env.dotenv_values")
def test_load_env_non_string_keys_values(mock_dotenv_values, mock_load_dotenv):
    mock_load_dotenv.return_value = True
    non_string_keys_values = {
        123: 456,
        True: False,
    }  # a dictionary with non-string keys and values
    mock_dotenv_values.return_value = non_string_keys_values
    with pytest.raises(ValueError):
        load_env()
