from functools import wraps
import pandas as pd  # type: ignore
import polars as pl
from typing import Callable, Union, Optional, List

from predictables.util.src._to_pl import to_pl_lf


def validate_column(
    func: Callable,
    data_frame: Union[pd.DataFrame, pl.DataFrame, pl.LazyFrame],
    column_name: Optional[str] = None,
):
    """
    Decorator to validate that a passed name represents a column in a DataFrame.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        col_name = _get_test_column_name(column_name, *args, **kwargs)
        df_ = _handle_df_types(data_frame)
        cols = _get_df_columns(df_)
        is_col_in_df, msg = _test_column_name_in_columns(col_name, cols)
        if is_col_in_df:
            return func(*args, **kwargs)
        else:
            raise ValueError(msg)

    return wrapper


def _get_test_column_name(column_name: Optional[str] = None, *args, **kwargs):
    is_col_found = False
    if (len(args) == 0) and ("column" not in kwargs) and (column_name not in kwargs):
        raise ValueError("No arguments passed to function.")

    if column_name is not None:
        col_name = column_name
        is_col_found = True
    else:
        for c in [
            "column",
            "col",
            "date_col",
            "date_column",
            "x_col",
            "x_column",
            "index_col",
            "index_column",
        ]:
            if c in kwargs:
                col_name = c
                is_col_found = True
                break

    if not is_col_found:
        raise ValueError("No column name found in function arguments.")

    return col_name


def _handle_df_types(df: Union[pd.DataFrame, pl.DataFrame, pl.LazyFrame]):
    if isinstance(df, pd.DataFrame):
        df_ = to_pl_lf(df)
    elif isinstance(df, pl.DataFrame):
        df_ = df.lazy()
    elif isinstance(df, pl.LazyFrame):
        df_ = df
    else:
        raise ValueError(f"Expected a DataFrame or LazyFrame, got {type(df)} instead.")
    return df_


def _get_df_columns(df: Union[pd.DataFrame, pl.DataFrame, pl.LazyFrame]):
    if isinstance(df, pd.DataFrame):
        return df.columns.tolist()
    elif isinstance(df, pl.DataFrame):
        return df.columns
    elif isinstance(df, pl.LazyFrame):
        return df.columns
    else:
        raise ValueError(f"Expected a DataFrame or LazyFrame, got {type(df)} instead.")


def _test_column_name_in_columns(col_name: str, cols: List[str]):
    if col_name not in cols:
        msg = f"Column {col_name} not found in DataFrame. "
        msg += "Please provide a valid column name:"
        for c in cols:
            msg += f"\n- {c}"
        return False, msg
    return True, None
