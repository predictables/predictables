def get_rc_params() -> dict:
    """
    Returns a dictionary of Matplotlib RC parameters for customizing plot styles.

    The function returns a dictionary of Matplotlib RC parameters that can be used to
    customize the style of Matplotlib plots. The parameters include font sizes, tick
    label sizes, legend font size, figure size, and figure DPI.

    Returns:
    --------
    dict
        A dictionary of Matplotlib RC parameters.
    """
    new_rc = {}

    new_rc["font.size"] = 12
    new_rc["axes.titlesize"] = 16
    new_rc["axes.labelsize"] = 14
    new_rc["xtick.labelsize"] = 14
    new_rc["ytick.labelsize"] = 14
    # new_rc['legend.fontsize'] = 14
    new_rc["figure.titlesize"] = 16

    # Set default figure size
    new_rc["figure.figsize"] = (17, 17)

    # Set default figure dpi
    new_rc["figure.dpi"] = 150

    return new_rc
