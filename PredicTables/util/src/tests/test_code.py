# import pytest
# from unittest.mock import mock_open, patch, MagicMock
# from .. import code_


# def test_read_file_code_success():
#     """Test that read_file_code returns the correct string."""
#     test_data = "Sample content of the file"
#     with patch("builtins.open", mock_open(read_data=test_data)) as mock_file:
#         result = code_.read_file_code("dummy_path")
#         mock_file.assert_called_once_with("dummy_path", "r")
#         assert result == test_data, f"Expected {test_data}, got {result}"
#     test_data2 = """def testing_a_real_looking_function():
#     this = "this"
#     is = "is"
#     a = "a"
#     real = "real"
#     looking = "looking"
#     function = "function"
#     return this, is, a, real, looking, function"""
#     with patch("builtins.open", mock_open(read_data=test_data2)) as mock_file:
#         result = code_.read_file_code("dummy_path")
#         mock_file.assert_called_once_with("dummy_path", "r")
#         assert result == test_data2, f"Expected {test_data2}, got {result}"


# def test_read_file_code_file_not_found():
#     """Test that read_file_code raises FileNotFoundError when file is not found."""
#     with patch("builtins.open", side_effect=FileNotFoundError):
#         with pytest.raises(FileNotFoundError):
#             code_.read_file_code("non_existent_file")


# def test_read_file_code_empty_file():
#     """Test that read_file_code returns an empty string when the file is empty."""
#     with patch("builtins.open", mock_open(read_data="")) as mock_file:
#         result = code_.read_file_code("empty_file")
#         mock_file.assert_called_once_with("empty_file", "r")
#         assert result == "", f"Expected empty string, got {result}"


# def test_get_functions_from_file_multiple_functions():
#     """Test that get_functions_from_file returns a list of functions from a file with multiple functions."""
#     mock_file_content = "def function1():\n    pass\ndef function2(param):\n    pass"
#     with patch("builtins.open", mock_open(read_data=mock_file_content)):
#         result = code_.get_functions_from_file("dummy_path")
#         assert result == [
#             "function1",
#             "function2",
#         ], f"Expected ['function1', 'function2'], got {result}"


# def test_get_functions_from_file_no_functions():
#     """Test that get_functions_from_file returns an empty list when there are no functions in the file."""
#     mock_file_content = "print('Hello World')"
#     with patch("builtins.open", mock_open(read_data=mock_file_content)):
#         result = code_.get_functions_from_file("dummy_path")
#         assert result == [], f"Expected [], got {result}"


# def test_get_functions_from_file_nested_functions():
#     """Test that get_functions_from_file returns a list of functions from a file with nested functions."""
#     mock_file_content = "def outer_function():\n    def inner_function():\n        pass"
#     with patch("builtins.open", mock_open(read_data=mock_file_content)):
#         result = code_.get_functions_from_file("dummy_path")
#         assert (
#             result == ["outer_function", "inner_function"]
#         ), f"Expected ['outer_function', 'inner_function'], got {result}. Note that 'inner_function' is nested inside 'outer_function'."


# def test_get_functions_from_file_unusual_definitions():
#     """Test that get_functions_from_file returns a list of functions from a file with unusual function definitions."""
#     mock_file_content = "def function(param1, param2=default):\n    pass"
#     with patch("builtins.open", mock_open(read_data=mock_file_content)):
#         result = code_.get_functions_from_file("dummy_path")
#         assert result == ["function"], f"Expected ['function'], got {result}"


# def test_get_functions_from_file_empty_file():
#     """Test that get_functions_from_file returns an empty list when the file is empty."""
#     mock_file_content = ""
#     with patch("builtins.open", mock_open(read_data=mock_file_content)):
#         result = code_.get_functions_from_file("dummy_path")
#         assert result == [], f"Expected [], got {result}"


# # def test_get_function_code_existing_function():
# #     mock_file_content = (
# #         "def function1():\n    print('Hello')\n\ndef function2():\n    pass"
# #     )
# #     with patch("builtins.open", mock_open(read_data=mock_file_content)):
# #         result = code_.get_function_code("function1", "dummy_path")
# #         expected_result = "print('Hello')"
# #         assert (
# #             result == expected_result
# #         ), f"Expected `{expected_result}`, got `{result}`"
# #         result = code_.get_function_code("function2", "dummy_path")
# #         expected_result = "pass"
# #         assert (
# #             result == expected_result
# #         ), f"Expected `{expected_result}`, got `{result}`"


# # def test_get_function_code_complex_arguments():
# #     """Test that get_function_code returns the correct code for a function with complex arguments."""
# #     mock_file_content = "def function(param1, param2=default):\n    pass"
# #     with patch("builtins.open", mock_open(read_data=mock_file_content)):
# #         result = code_.get_function_code("function", "dummy_path")
# #         assert result == "\n    pass", f"Expected \n    pass, got {result}"


# # def test_get_function_code_non_existent_function():
# #     """Test that get_function_code raises an ValueError when the function does not exist."""
# #     mock_file_content = "def function1():\n    pass"
# #     with patch("builtins.open", mock_open(read_data=mock_file_content)):
# #         with pytest.raises(ValueError):
# #             code_.get_function_code("function2", "dummy_path")


# # def test_get_function_code_nested_functions():
# #     """Test that get_function_code returns the correct code for a nested function."""
# #     mock_file_content = "def outer_function():\n    def inner_function():\n        pass\n    return inner_function"
# #     with patch("builtins.open", mock_open(read_data=mock_file_content)):
# #         result = code_.get_function_code("outer_function", "dummy_path")
# #         assert "def inner_function()" in result


# # def test_get_function_code_no_functions():
# #     mock_file_content = "print('Hello World')"
# #     with patch("builtins.open", mock_open(read_data=mock_file_content)):
# #         with pytest.raises(ValueError):
# #             code_.get_function_code("function1", "dummy_path")


# def test_get_files_from_folder_all_files():
#     """Test that get_files_from_folder returns all files when no type is specified."""
#     mock_files = ["file1.txt", "file2.py", "file3.jpg"]
#     with patch("os.listdir", return_value=mock_files):
#         result = code_.get_files_from_folder("dummy_path")
#         assert set(result) == set(mock_files), f"Expected {mock_files}, got {result}"


# def test_get_files_from_folder_specific_type():
#     """Test that get_files_from_folder returns only files of the specified type."""
#     mock_files = ["file1.txt", "file2.py", "file3.jpg"]
#     with patch("os.listdir", return_value=mock_files):
#         result = code_.get_files_from_folder("dummy_path", "py")
#         assert result == ["file2.py"], f"Expected ['file2.py'], got {result}"


# def test_get_files_from_folder_no_matching_type():
#     """Test that get_files_from_folder returns an empty list when there are no files of the specified type."""
#     mock_files = ["file1.txt", "file2.txt", "file3.jpg"]
#     with patch("os.listdir", return_value=mock_files):
#         result = code_.get_files_from_folder("dummy_path", "py")
#         assert result == [], f"Expected [], got {result}"


# def test_get_files_from_folder_empty_folder():
#     """Test that get_files_from_folder returns an empty list when the folder is empty."""
#     with patch("os.listdir", return_value=[]):
#         result = code_.get_files_from_folder("dummy_path")
#         assert result == [], f"Expected [], got {result}"


# def test_get_files_from_folder_non_existent_folder():
#     """Test that get_files_from_folder raises a FileNotFoundError when the folder does not exist."""
#     with patch("os.listdir", side_effect=FileNotFoundError):
#         with pytest.raises(FileNotFoundError):
#             code_.get_files_from_folder("non_existent_path")


# def test_copy_folder_code_multiple_files():
#     mock_files = ["file1.py", "file2.py"]
#     mock_file_contents = {
#         "dummy_path/file1.py": "print('Hello')",
#         "dummy_path/file2.py": "print('World')",
#     }

#     def mock_open_multiple_files(file, *args, **kwargs):
#         if file in mock_file_contents:
#             return mock_open(read_data=mock_file_contents[file]).return_value
#         return mock_open(read_data="").return_value

#     with patch("os.listdir", return_value=mock_files), patch(
#         "builtins.open",
#         new_callable=lambda: MagicMock(side_effect=mock_open_multiple_files),
#     ), patch("pyperclip.copy") as mock_clipboard:
#         code_.copy_folder_code("dummy_path", "py")
#         mock_clipboard.assert_called_once_with("print('Hello')\nprint('World')")


# def test_copy_folder_code_no_files_of_type():
#     """Test that copy_folder_code copies nothing when there are no files of the specified type."""
#     with patch("os.listdir", return_value=[]):
#         with patch("pyperclip.copy") as mock_clipboard:
#             code_.copy_folder_code("dummy_path", "py")
#             mock_clipboard.assert_called_once_with("")


# def test_copy_folder_code_empty_folder():
#     with patch("os.listdir", return_value=[]):
#         with patch("pyperclip.copy") as mock_clipboard:
#             code_.copy_folder_code("dummy_path", "py")
#             mock_clipboard.assert_called_once_with("")


# def test_copy_folder_code_non_existent_folder():
#     with patch("os.listdir", side_effect=FileNotFoundError):
#         with patch("pyperclip.copy") as mock_clipboard:
#             with pytest.raises(FileNotFoundError):
#                 code_.copy_folder_code("non_existent_path", "py")
