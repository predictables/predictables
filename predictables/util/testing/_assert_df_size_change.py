from typing import Union

import pandas as pd
import polars as pl

from predictables.util.src._to_pl import to_pl_df


def assert_df(func, row: int = 0, col: int = 0):
    """
    This is a decorator that asserts that the number of rows and columns in a dataframe have changed by the expected amount. It is used to ensure that a function that is supposed to add or remove rows and columns is working as expected, and no rows or columns are 'accidentally' added or removed.

    Parameters
    ----------
    func : function
        The function to be tested.
    row : int, optional
        The expected change in rows, by default 0, indicating no intended change.
    col : int, optional
        The expected change in columns, by default 0, indicating no intended change.

    Returns
    -------
    function
        The decorated function.

    Raises
    ------
    AssertionError
        If the number of rows or columns have changed by an unexpected amount -- eg if the number of rows or columns have changed by a number different from the passed `row` or `col` parameters.

    Examples
    --------
    >>> import pandas as pd
    >>> @assert_df(row=-1)
    ... def drop_first_row(df: pd.DataFrame):
    ...     return df.drop(index=0)
    >>> df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
    >>> print(drop_first_row(df))
        a  b
    1   2  5
    2   3  6
    """

    def wrapper(*args, **kwargs):
        df = to_pl_df(args[0])
        result = func(*args, **kwargs)
        assert_df_size_change(df, to_pl_df(result), row, col)
        return result

    return wrapper


def assert_df_size_change(
    df: Union[pd.DataFrame, pl.DataFrame, pl.LazyFrame],
    df1: Union[pd.DataFrame, pl.DataFrame, pl.LazyFrame],
    row: int = 0,
    col: int = 0,
):
    """
    This is a testing function that asserts that the number of rows and columns in a dataframe have changed by the expected amount. It is used to ensure that a function that is supposed to add or remove rows and columns is working as expected, and no rows or columns are 'accidentally' added or removed.

    Parameters
    ----------
    df : Union[pd.DataFrame, pl.DataFrame, pl.LazyFrame]
        The dataframe before the function is applied.
    df1 : Union[pd.DataFrame, pl.DataFrame, pl.LazyFrame]
        The dataframe after the function is applied.
    row : int, optional
        The expected change in rows, by default 0, indicating no intended change.
    col : int, optional
        The expected change in columns, by default 0, indicating no intended change.

    Returns
    -------
    bool
        True if the number of rows and columns have changed by the expected amount, False otherwise.

    Raises
    ------
    AssertionError
        If the number of rows or columns have changed by an unexpected amount.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
    >>> df1 = df.copy().drop(columns='b')
    >>> assert_df_size_change(df, df1, col=-1)
    True

    >>> df2 = df.copy().drop(index=0)
    >>> assert_df_size_change(df, df2, row=-1)
    True
    """
    df = to_pl_df(df)
    df1 = to_pl_df(df1)

    try:
        assert (
            df.shape[0] + row == df1.shape[0]
        ), f"rows before: {df.shape[0]}\nrows after: {df1.shape[0]}\nexpected row change: {row}\nactual row change: {df1.shape[0] - df.shape[0]}"
        assert (
            df.shape[1] + col == df1.shape[1]
        ), f"columns before: {df.shape[1]}\ncolumns after: {df1.shape[1]}\nexpected column change: {col}\nactual column change: {df1.shape[1] - df.shape[1]}"

        if col > 0:
            if col > 10:
                print(f"Added {col} columns.")
            else:
                print(f"Added columns:\n\n{set(df1.columns) - set(df.columns)}")

        return True
    except AssertionError as e:
        assert (
            df.shape[0] + row == df1.shape[0]
        ), f"rows before: {df.shape[0]}\nrows after: {df1.shape[0]}\nexpected row change: {row}\nactual row change: {df1.shape[0] - df.shape[0]}\n\n{e}"

        assert (
            df.shape[1] + col == df1.shape[1]
        ), f"columns before: {df.shape[1]}\ncolumns after: {df1.shape[1]}\nexpected column change: {col}\nactual column change: {df1.shape[1] - df.shape[1]}\n\nLikely due to: {set(df1.columns) - set(df.columns) if df1.shape[1] > df.shape[1] else set(df.columns) - set(df1.columns)}\n\n{e}"
