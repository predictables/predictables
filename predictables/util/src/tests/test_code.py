from unittest.mock import mock_open, patch

import pytest

from predictables.util.src._code import (  # get_function_docstring,
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


# # Testing read_file_code function
# @pytest.mark.parametrize(
#     "file_name, expected_output",
#     [
#         ("valid_python.py", mock_file_contents["valid_python.py"]),
#         (
#             "valid_python_2_docstrings.py",
#             mock_file_contents["valid_python_2_docstrings.py"],
#         ),
# (
#     "valid_python_3_func_2_docstrings.py",
#     mock_file_contents["valid_python_3_func_2_docstrings.py"],
# ),
# ("empty_file.py", mock_file_contents["empty_file.py"]),
# ("non_python.txt", mock_file_contents["non_python.txt"]),
#     ],
# )
# def test_read_file_code(file_name, expected_output):
#     with patch("builtins.open", mock_open(read_data=mock_file_contents[file_name])):
#         assert (
#             read_file_code(file_name) == expected_output
#         ), f"Expected output: {expected_output}, Actual output: {read_file_code(file_name)}"


@pytest.mark.parametrize(
    "file_name", [("non_existent_file.py"), ("incorrect/path/to/file.py")]
)
def test_read_file_code_invalid(file_name):
    with patch(
        "builtins.open", mock_open(read_data=mock_file_contents.get(file_name, ""))
    ):
        assert (
            read_file_code(file_name) == "PYTEST_FILE_NOT_FOUND"
        ), f"Expected output: PYTEST_FILE_NOT_FOUND, Actual output: {read_file_code(file_name)}"


# Testing get_functions_from_file function
@pytest.mark.parametrize(
    "file_name, expected_output",
    [
        # ("valid_python.py", ["example_function", "another_function"]),
        # ("valid_python_2_docstrings.py", ["example_function", "another_function"]),
        # (
        #     "valid_python_3_func_2_docstrings.py",
        #     ["example_function", "a_second_function_no_ds", "a_third_function_with_ds"],
        # ),
        ("empty_file.py", []),
        ("no_functions.py", []),
    ],
)
def test_get_functions_from_file(file_name, expected_output):
    with patch("builtins.open", mock_open(read_data=mock_file_contents[file_name])):
        assert (
            get_functions_from_file(file_name) == expected_output
        ), f"Expected output: {expected_output}, Actual output: {get_functions_from_file(file_name)}"


# Testing get_function_docstring function
# @pytest.mark.parametrize(
#     "file_name, function_name, expected_output",
#     [
#         ("valid_python.py", "example_function", "Example function docstring."),
#         ("valid_python.py", "non_existent_function", ""),
# (
#     "valid_python_2_docstrings.py",
#     "example_function",
#     "Example function docstring.",
# ),
# (
#     "valid_python_2_docstrings.py",
#     "another_function",
#     "Example function docstring 2.",
# ),
# (
#     "valid_python_3_func_2_docstrings.py",
#     "example_function",
#     "Example function docstring.",
# ),
# (
#     "valid_python_3_func_2_docstrings.py",
#     "a_second_function_no_ds",
#     "",
# ),
# (
#     "valid_python_3_func_2_docstrings.py",
#     "a_third_function_with_ds",
#     "Example function docstring 2.",
# ),
#         ("empty_file.py", "example_function", ""),
#         ("no_functions.py", "example_function", ""),
#     ],
# )
# def test_get_function_docstring(file_name, function_name, expected_output):
#     """
#     Test get_function_docstring function. Ensure correct docstring is returned for
#     valid function, and IndexError is raised for invalid function.
#     """
#     with patch("builtins.open", mock_open(read_data=mock_file_contents[file_name])):
#         assert (
#             get_function_docstring(function_name, file_name) == expected_output
#         ), f"Expected output: {expected_output}, Actual output: {get_function_docstring(function_name, file_name)}"


# Testing get_files_from_folder function
@pytest.mark.parametrize(
    "folder_contents, file_type, expected_output",
    [
        (
            [
                "valid_python.py",
                "valid_python_2_docstrings.py",
                "valid_python_3_func_2_docstrings.py",
                "empty_file.py",
                "no_functions.py",
            ],
            None,
            [
                "valid_python.py",
                "valid_python_2_docstrings.py",
                "valid_python_3_func_2_docstrings.py",
                "empty_file.py",
                "no_functions.py",
            ],
        ),
        ([], None, []),
        (
            [
                "valid_python.py",
                "valid_python_2_docstrings.py",
                "valid_python_3_func_2_docstrings.py",
                "non_python.txt",
            ],
            "py",
            [
                "valid_python.py",
                "valid_python_2_docstrings.py",
                "valid_python_3_func_2_docstrings.py",
            ],
        ),
        (
            [
                "valid_python.py",
                "valid_python_2_docstrings.py",
                "valid_python_3_func_2_docstrings.py",
                "non_python.txt",
            ],
            "txt",
            ["non_python.txt"],
        ),
        (
            [
                "valid_python.py",
                "valid_python_2_docstrings.py",
                "valid_python_3_func_2_docstrings.py",
                "non_python.txt",
            ],
            "md",
            [],
        ),
    ],
)
def test_get_files_from_folder(folder_contents, file_type, expected_output):
    with patch("os.listdir", return_value=folder_contents):
        assert (
            sorted(get_files_from_folder("some_folder", file_type))
            == sorted(expected_output)
        ), f"Expected output: {expected_output}, Actual output: {get_files_from_folder('some_folder', file_type)}"
