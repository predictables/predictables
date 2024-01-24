def plot_label(s: str) -> str:
    """
    Convert a string to a formatted plot label.

    Parameters
    ----------
    s : str
        The string to convert

    Returns
    -------
    str
        The converted string
    """
    if s == "":
        return ""
    else:
        s = s.replace("_", " ").title()
        return f"[{s}]" if s[0] != "[" else s
