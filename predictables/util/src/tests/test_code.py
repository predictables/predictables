from unittest.mock import mock_open, patch

import pytest

from predictables.util.src._code import (
    get_files_from_folder,
    get_functions_from_file,
    read_file_code,
)

# Mock file contents for testing
mock_file_contents = {
    "valid_python.py": (
        "def example_function():\n"
        '    """Example function docstring."""\n'
        "    pass\n\n"
        "def another_function():\n"
        "    pass\n"
    ),
    "valid_python_2_docstrings.py": (
        "def example_function():\n"
        '    """Example function docstring."""\n'
        "    pass\n\n"
        "def another_function():\n"
        "    '''Example function docstring 2.'''\n"
        "    pass\n"
    ),
    "valid_python_3_func_2_docstrings.py": (
        "def example_function():\n"
        '    """Example function docstring."""\n'
        "    pass\n\n"
        "def a_second_function_no_ds():\n"
        "    pass\n\n"
        "def a_third_function_with_ds():\n"
        '    """Example function docstring 2."""\n'
        "    pass\n"
    ),
    "empty_file.py": "",
    "no_functions.py": "# Just a comment\nx = 5\n",
    "non_python.txt": "This is not a python file.\n",
}


@pytest.mark.parametrize(
    "file_name", [("non_existent_file.py"), ("incorrect/path/to/file.py")]
)
def test_read_file_code_invalid(file_name: str):
    with patch(
        "builtins.open", mock_open(read_data=mock_file_contents.get(file_name, ""))
    ):
        assert (
            read_file_code(file_name) == "PYTEST_FILE_NOT_FOUND"
        ), f"Expected output: PYTEST_FILE_NOT_FOUND, Actual output: {read_file_code(file_name)}"


# Testing get_functions_from_file function
@pytest.mark.parametrize(
    "file_name, expected_output", [("empty_file.py", []), ("no_functions.py", [])]
)
def test_get_functions_from_file(file_name: str, expected_output: list):
    with patch("builtins.open", mock_open(read_data=mock_file_contents[file_name])):
        assert (
            get_functions_from_file(file_name) == expected_output
        ), f"Expected output: {expected_output}, Actual output: {get_functions_from_file(file_name)}"
