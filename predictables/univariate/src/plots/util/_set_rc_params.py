from ._get_rc_params import get_rc_params


def set_rc_params(rc_params: dict) -> dict:
    """Set Matplotlib RC parameters for customizing plot styles.

    The function sets Matplotlib RC parameters for customizing the style of Matplotlib
    plots. The parameters include font sizes, tick label sizes, legend font size,
    figure size, and figure DPI. The parameters are obtained from the `get_rc_params`
    function.

    Parameters
    ----------
    rc_params : dict
        A dictionary of Matplotlib RC parameters to be updated.

    Returns
    -------
    dict
        A dictionary of Matplotlib RC parameters with the updated values.
    """
    for k, v in get_rc_params().items():
        rc_params[k] = v
    return rc_params
