import re


def fmt_col_name(col_name: str) -> str:
    """
    Formats a column name to be used as an attribute name within the
    UnivariateAnalysis class. Removes non-alphanumeric characters, replaces
    spaces and special characters with underscores, and ensures the resulting
    string does not end with an underscore.

    Parameters
    ----------
    col_name : str
        The original column name to format.

    Returns
    -------
    str
        The formatted column name suitable for use as a Python attribute,
        not ending with an underscore.

    Examples
    --------
    >>> _fmt_col_name("Total Revenue - 2020")
    'total_revenue_2020'
    >>> _fmt_col_name("Cost/Unit_")
    'cost_unit'
    """
    col_name = re.sub(r"\W+", "_", col_name)  # Replace all non-word characters with _
    col_name = re.sub(
        r"__+", "_", col_name
    )  # Normalize multiple underscores to single _
    col_name = col_name.lower()  # Convert to lowercase
    if col_name.endswith("_"):
        col_name = col_name[:-1]  # Remove trailing underscore if present
    return col_name
