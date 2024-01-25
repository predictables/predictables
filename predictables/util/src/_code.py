import os
from typing import Optional


def read_file_code(filepath: str) -> str:
    """
    Takes a file path and returns the code as a string. Assumes that the file uses consistent indentation (spaces or tabs).

    Parameters
    ----------
    filepath : str
        The file path to the file to read.

    Returns
    -------
    str
        The code in the file.

    Examples
    --------
    >>> # Will use the print function for prettier output
    >>> print(read_file_code("predictables/util/src/_code.py")) # doctest: skip
    # Path: predictables/util/src/_code.py
    import os
    from typing import Optional

    def read_file_code(filepath: str) -> str:
        ...(other code)...
    """
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
    ['read_file_code', 'get_functions_from_file', 'get_function_code', 'get_function_docstring', 'get_files_from_folder', 'copy_folder_code']

    """
    code = read_file_code(filepath)
    return [
        line.split("def ")[1].split("(")[0]
        for line in code.split("\n")
        if "def " in line
    ]


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
    """
    Takes a function name and file path and returns the docstring as a string. Assumes that the file uses consistent indentation (spaces or tabs).

    Parameters
    ----------
    function_name : str
        The name of the function to get the docstring for.
    filepath : str
        The file path to the file to read.

    Returns
    -------
    str
        The docstring for the function.

    Examples
    --------
    >>> # Will use the print function for prettier output
    >>> print(get_function_docstring("get_functions_from_file", "predictables/util/src/_code.py"))
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
    ['read_file_code', 'get_functions_from_file', 'get_function_code', 'get_function_docstring', 'get_files_from_folder', 'copy_folder_code']

    >>> # This is just the docstring, copy/pasted from the function above
    """
    code = read_file_code(filepath)
    code = code.split(f"def {function_name}(")[0]
    code = code.split('"""')[1]
    return code


def get_files_from_folder(folder_path: str, file_type: Optional[str] = None) -> list:
    """
    Takes a folder path and returns a list of files in that folder. If a file type is specified, only files of that type will be returned, otherwise all files will be returned.

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
        if file_type is None:
            files.append(file)
        elif file.endswith(f".{file_type}"):
            files.append(file)
    return files
