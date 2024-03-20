from __future__ import annotations


def rolling_op_column_name(
    op: str,
    value_col: str,
    category_col: str | None = None,
    offset: int = 0,
    window: int = 0,
) -> str:
    """Generate the column name from a rolling operation.

    Parameters
    ----------
    op : str
        The rolling operation, such as ROLLING_SUM or ROLLING_MEAN.
    value_col : str
        The name of the column to be used for the rolling operation (eg
        the values that will be summed or averaged).
    category_col : str, default None
        If provided, these are the columns that the aggregation is grouped by.
    offset : int, default 0
        The offset for the rolling operation. The latest date included is the
        date in a date column minus this many days. Default is 0.
    window : int, default 0
        The window for the rolling operation. This is the number of days
        included in the rolling operation. The earliest date included is the
        date in a date column minus the sum of the offset and window.
        Default is 0.

    Returns
    -------
    str
        The column name for the rolling operation.
    """
    cat_chunk = f"[{category_col}]" if category_col else "[ALL]"
    lag_chunk = f"lag:{offset}"
    win_chunk = f"win:{window}"
    return f"{op}({value_col}{cat_chunk})[{lag_chunk}/{win_chunk}]"
