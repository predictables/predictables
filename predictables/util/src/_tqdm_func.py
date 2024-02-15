import os
from typing import Any, Iterable, Optional

from dotenv import load_dotenv
from tqdm import tqdm as _tqdm  # type: ignore
from tqdm.notebook import tqdm as _tqdm_notebook  # type: ignore


def identidy_function(x: Any) -> Any:
    return x


def tqdm(
    x: Iterable,
    desc: Optional[str] = None,
    enable: Optional[bool] = None,
    disable: Optional[bool] = None,
    notebook: Optional[bool] = None,
    nb: Optional[bool] = None,
) -> Iterable:
    """
    Wrapper for tqdm that can be enabled or disabled by setting the
    TQDM_ENABLE environment variable to "true" or "false". If the
    environment variable is not set, the default is to enable
    tqdm.

    Parameters
    ----------
    x : Any
        Any iterable to be wrapped by tqdm.
    desc : str, optional
        Description to be displayed by tqdm.
    enable : bool, optional
        If True, enable tqdm. If False, disable tqdm. If both enable
        and disable are set, enable takes precedence.
    disable : bool, optional
        If True, disable tqdm. If False, enable tqdm. If both enable
        and disable are set, enable takes precedence.
    notebook : bool, optional
        If True, use tqdm.notebook.tqdm. If False, use tqdm.tqdm. If
        both notebook and nb are set, notebook takes precedence.
    nb : bool, optional
        If True, use tqdm.notebook.tqdm. If False, use tqdm.tqdm. If
        both notebook and nb are set, notebook takes precedence.

    Returns
    -------
    Callable
        A function that wraps the input iterable with tqdm if it is enabled, or
        returns the input iterable unchanged if it is disabled.
    """
    load_dotenv()

    # Handle description (if provided)
    desc_dict = {"desc": desc} if desc is not None else {}

    # Handle enable, disable, notebook, and nb (if any)
    if enable is not None:
        tqdm_enable = (
            enable
            if isinstance(enable, bool)
            else os.environ.get("TQDM_ENABLE") == "true"
        )
    elif disable is not None:
        tqdm_enable = (
            not disable
            if isinstance(disable, bool)
            else os.environ.get("TQDM_ENABLE") == "true"
        )
    else:
        tqdm_enable = os.environ.get("TQDM_ENABLE") == "true"

    if notebook is not None:
        tqdm_nb = (
            notebook
            if isinstance(notebook, bool)
            else os.environ.get("TQDM_NOTEBOOK") == "true"
        )
    elif nb is not None:
        tqdm_nb = (
            nb if isinstance(nb, bool) else os.environ.get("TQDM_NOTEBOOK") == "true"
        )
    else:
        tqdm_nb = os.environ.get("TQDM_NOTEBOOK") == "true"

    # Return the appropriate function
    if tqdm_nb:
        return _tqdm_notebook(x, **desc_dict) if tqdm_enable else identidy_function(x)
    else:
        return _tqdm(x, **desc_dict) if tqdm_enable else identidy_function(x)
