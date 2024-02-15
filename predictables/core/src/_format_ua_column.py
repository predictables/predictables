import polars as pl


def results_tbl_expr(col, formatted_name, formatting_str) -> pl.Expr:
    """
    Create an expression to format a column in a results table.

    Parameters
    ----------
    col : pl.Expr
        The column to format.
    formatted_name : str
        The name of the formatted column.
    formatting_str : str
        The formatting string.

    Returns
    -------
    pl.Expr
        The formatted column expression.
    """
    return (
        pl.format(formatting_str, pl.col(col))
        .cast(pl.Utf8)
        .alias(formatted_name)
    )
