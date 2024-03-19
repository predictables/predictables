from __future__ import annotations


def get_rc_params() -> dict[str, float | int | tuple[int, int] | str]:
    """Return a dictionary of Matplotlib RC parameters for customizing plot styles.

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
        "figure.figsize": (7, 7),
        "figure.dpi": 150,
    }
