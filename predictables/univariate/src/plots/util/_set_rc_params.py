from ._get_rc_params import get_rc_params


def set_rc_params(rcParams) -> dict:
    """
    Sets Matplotlib RC parameters for customizing plot styles.

    The function sets Matplotlib RC parameters for customizing the style of Matplotlib
    plots. The parameters include font sizes, tick label sizes, legend font size,
    figure size, and figure DPI. The parameters are obtained from the `get_rc_params`
    function.

    Parameters:
    -----------
    rcParams : dict
        A dictionary of Matplotlib RC parameters to be updated.

    Returns:
    --------
    dict
        A dictionary of Matplotlib RC parameters with the updated values.
    """
    for k, v in get_rc_params().items():
        rcParams[k] = v
    return rcParams
