import os
from typing import Any, Callable

from dotenv import load_dotenv
from tqdm import tqdm as _tqdm  # type: ignore
from tqdm.notebook import tqdm as _tqdm_notebook  # type: ignore


def identidy_function(x: Any) -> Any:
    return x


def tqdm(**kwargs: Any) -> Callable:
    """
    Wrapper for tqdm that can be enabled or disabled by setting the TQDM_ENABLE environment variable to "true" or "false".
    If the environment variable is not set, the default is to enable tqdm.

    Parameters
    ----------
    desc : str, optional
        Description to be displayed by tqdm.
    enable : bool, optional
        If True, enable tqdm. If False, disable tqdm. If both enable and disable are set, enable takes precedence.
    disable : bool, optional
        If True, disable tqdm. If False, enable tqdm. If both enable and disable are set, enable takes precedence.
    notebook : bool, optional
        If True, use tqdm.notebook.tqdm. If False, use tqdm.tqdm. If both notebook and nb are set, notebook takes precedence.
    nb : bool, optional
        If True, use tqdm.notebook.tqdm. If False, use tqdm.tqdm. If both notebook and nb are set, notebook takes precedence.

    Returns
    -------
    Callable
        A function that wraps the input iterable with tqdm if it is enabled, or returns the input iterable unchanged if it is disabled.
    """
    load_dotenv()

    # Handle description (if provided)
    desc_dict = {"desc": kwargs["desc"]} if "desc" in kwargs else {}

    # Handle enable, disable, notebook, and nb (if any)
    if "enable" in kwargs:
        tqdm_enable = kwargs["enable"] if isinstance(kwargs["enable"], bool) else True
    elif "disable" in kwargs:
        tqdm_enable = (
            not kwargs["disable"] if isinstance(kwargs["disable"], bool) else True
        )
    else:
        tqdm_enable = os.environ.get("TQDM_ENABLE") == "true"

    if "notebook" in kwargs:
        tqdm_nb = kwargs["notebook"] if isinstance(kwargs["notebook"], bool) else True
    elif "nb" in kwargs:
        tqdm_nb = kwargs["nb"] if isinstance(kwargs["nb"], bool) else True
    else:
        tqdm_nb = os.environ.get("TQDM_NOTEBOOK") == "true"

    # Return the appropriate function
    return (
        (lambda x: _tqdm_notebook(x, **desc_dict) if tqdm_nb else _tqdm(x, **desc_dict))
        if tqdm_enable
        else identidy_function
    )
