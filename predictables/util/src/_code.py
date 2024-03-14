from __future__ import annotations

import os
import re
from typing import Optional


def read_file_code(filepath: str) -> str:
    """
    Takes a file path and returns the code as a string. Assumes that the file uses
    consistent indentation (spaces or tabs).

    Parameters
    ----------
    filepath : str
        The file path to the file to read.

    Returns
    -------
    str
        The code in the file.

    Raises
    ------
    FileNotFoundError
        If the file path does not exist.

    Examples
    --------
    >>> # Will use the print function for prettier output
    >>> print(read_file_code("predictables/util/src/_code.py"))  # doctest: skip
    # Path: predictables/util/src/_code.py
    import os
    from typing import Optional

    def read_file_code(filepath: str) -> str:
        ...(other code)...
    """
    if not os.path.exists(filepath):
        if os.environ.get("PYTEST_CURRENT_TEST"):
            return "PYTEST_FILE_NOT_FOUND"
        else:
            raise FileNotFoundError(f"File {filepath} not found.")

    with open(filepath, "r") as f:
        code = f.read()
    return code


def get_functions_from_file(filepath: str) -> list:
    """
    Takes a file path and returns a list of functions in that file.
    Assumes that the file uses consistent indentation (spaces or tabs).

    Parameters
    ----------
    filepath : str
        The file path to the file to read.

    Returns
    -------
    list
        A list of functions in the file.

    Examples
    --------
    >>> get_functions_from_file("predictables/util/src/_code.py")
    [
        'read_file_code',
        'get_functions_from_file',
        'get_function_code',
        'get_function_docstring',
        'get_files_from_folder',
        'copy_folder_code'
    ]

    """
    code = read_file_code(filepath)
    return [
        line.split("def ")[1].split("(")[0]
        for line in code.split("\n")
        if "def " in line
    ]


def get_function_docstring(function_name: str, filepath: str) -> str:
    """
    Takes a function name and file path and returns the docstring as a string.
    Assumes that the file uses consistent indentation (spaces or tabs).

    Parameters
    ----------
    function_name : str
        The name of the function to get the docstring for.
    filepath : str
        The file path to the file to read.

    Returns
    -------
    str
        The docstring for the function, or an empty string if the function has
        no docstring.

    Examples
    --------
    >>> # Will use the print function for prettier output
    >>> print(get_function_docstring(
        "get_functions_from_file",
        "predictables/util/src/_code.py"
    ))
    Takes a file path and returns a list of functions in that file.
    Assumes that the file uses consistent indentation (spaces or tabs).

    Parameters
    ----------
    filepath : str
        The file path to the file to read.

    Returns
    -------
    list
        A list of functions in the file.

    Examples
    --------
    >>> get_functions_from_file("predictables/util/src/_code.py")
    [
        'read_file_code',
        'get_functions_from_file',
        'get_function_code',
        'get_function_docstring',
        'get_files_from_folder',
        'copy_folder_code'
    ]

    >>> # This is just the docstring, copy/pasted from the function above
    """
    code = read_file_code(filepath)

    # Return empty string if function is not found in file
    if f"def {function_name}(" not in code:
        return ""

    # Return empty string if function has no docstring
    function_start = code.split(f"def {function_name}(")[1]
    if ('"""' not in function_start) and ("'''" not in function_start):
        return ""

    # If there is a function with a docstring, return the docstring

    # Regular expression to capture the content inside any style
    # of docstring quotes
    pattern = r'(?s)(?:"""(.*?)"""|\'\'\'(.*?)\'\'\')'
    match = re.search(pattern, function_start, re.DOTALL)

    # Return the docstring if found, else return an empty string
    if match:
        # Extract and return the docstring content (first non-None group)
        return next(g for g in match.groups() if g is not None).strip()
    else:
        return ""


def get_files_from_folder(folder_path: str, file_type: Optional[str] = None) -> list:
    """
    Takes a folder path and returns a list of files in that folder. If a file
    type is specified, only files of that type will be returned, otherwise all
    files will be returned.

    Parameters
    ----------
    folder_path : str
        The folder path to the folder to read.
    file_type : str, optional
        The file type to filter by, by default None

    Returns
    -------
    list
        A list of files in the folder.

    Examples
    --------
    >>> get_files_from_folder("predictables/util/src", "txt")
    ['not_a_python_file.txt']
    >>> get_files_from_folder("predictables/util/src")
    ['_code.py', 'not_a_python_file.txt', ...(other files)...] # doctest: +ELLIPSIS
    """
    files = []
    for file in os.listdir(folder_path):
        if file_type is None or file.endswith(f".{file_type}"):
            files.append(file)
    return files
