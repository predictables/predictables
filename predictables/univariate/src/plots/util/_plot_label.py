def plot_label(s: str, incl_bracket: bool = True) -> str:
    """
    Convert a string to a formatted plot label.

    Parameters
    ----------
    s : str
        The string to convert
    incl_bracket : bool, optional
        Whether to include the square brackets around the string.
        Default is True.

    Returns
    -------
    str
        The converted string
    """
    if s == "":
        return ""
    else:
        s = s.replace("_", " ").title()
        if incl_bracket:
            return f"[{s}]" if s[0] != "[" else s
        else:
            return s
