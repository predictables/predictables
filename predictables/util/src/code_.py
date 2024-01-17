import os
import re

import pyperclip


def read_file_code(filepath: str) -> str:
    """Takes a file path and returns the code as a string."""
    with open(filepath, "r") as f:
        code = f.read()
    return code


def get_functions_from_file(filepath: str) -> list:
    """Takes a file path and returns a list of functions in that file."""
    code = read_file_code(filepath)
    functions = []
    for line in code.split("\n"):
        if "def " in line:
            functions.append(line.split("def ")[1].split("(")[0])
    return functions


# def get_function_code(function_name: str, filepath: str) -> str:
#     code = read_file_code(filepath)

#     # Regular expression to capture function code
#     # Assumes that the file uses consistent indentation (spaces or tabs)
#     pattern = rf"def {function_name}\(.*?\):((?:\n(?:    |\t).*)*)"

#     match = re.search(pattern, code, re.DOTALL)
#     if not match:
#         raise ValueError(f"Function {function_name} not found in file.")

#     function_code = match.group(1).strip()
#     return function_code


def get_function_docstring(function_name: str, filepath: str) -> str:
    """Takes a function name and file path and returns the docstring as a string."""
    code = read_file_code(filepath)
    code = code.split(f"def {function_name}(")[0]
    code = code.split('"""')[1]
    return code


def get_files_from_folder(folder_path: str, file_type: str = None) -> list:
    """Takes a folder path and returns a list of files in that folder."""
    files = []
    for file in os.listdir(folder_path):
        if file_type is None:
            files.append(file)
        elif file.endswith(f".{file_type}"):
            files.append(file)
    return files


def copy_folder_code(folder_path: str, file_type: str) -> None:
    """Takes a folder path and copies the code as a string to the clipboard."""
    files = get_files_from_folder(folder_path, file_type)
    code = ""
    for i, file in enumerate(files):
        if i > 0:
            code += "\n"
        code += read_file_code(os.path.join(folder_path, file))
    pyperclip.copy(code)
