import re
from predictables.util.src._fmt_col_name import fmt_col_name


def col_name_for_report(col_name: str) -> str:
    """
    Formats a column name to be used as a title in a report. Replaces underscores
    with spaces and capitalizes the first letter of each word.

    Parameters
    ----------
    col_name : str
        The original column name to format.

    Returns
    -------
    str
        The formatted column name suitable for use as a title in a report.

    Examples
    --------
    >>> _col_name_for_report("total_revenue_2020")
    'Total Revenue 2020'
    >>> _col_name_for_report("cost_unit")
    'Cost Unit'
    """
    if not isinstance(col_name, str):
        raise ValueError(
            f"Invalid value {col_name} for column name. Expected a string, but got {type(col_name)}."
        )
    return (
        re.sub(r"_+", " ", fmt_col_name(col_name))
        .title()
        .replace("Log", "log")
        .replace("1P", "1p")
    )
