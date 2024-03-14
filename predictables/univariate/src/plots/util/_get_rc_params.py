from __future__ import annotations

from typing import Any, Dict


def get_rc_params() -> Dict[str, Any]:
    """
    Returns a dictionary of Matplotlib RC parameters for customizing plot styles.

    The function returns a dictionary of Matplotlib RC parameters that can be used to
    customize the style of Matplotlib plots. The parameters include font sizes, tick
    label sizes, legend font size, figure size, and figure DPI.

    Returns
    -------
    dict
        A dictionary of Matplotlib RC parameters.
    """
    return {
        "font.size": 12,
        "axes.titlesize": 16,
        "axes.labelsize": 14,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "figure.titlesize": 16,
        "figure.figsize": (7, 7),  # type: ignore
        "figure.dpi": 150,
    }
