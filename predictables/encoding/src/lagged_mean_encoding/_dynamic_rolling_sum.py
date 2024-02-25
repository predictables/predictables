"""
This module provides functionality for calculating dynamic rolling sums on time series data
within a Polars LazyFrame. It is designed to support complex scenarios, including the handling
of categorical variables and custom rolling window parameters. The main function, `dynamic_rolling_sum`,
calculates the rolling sum for a specified column over a defined period, with an optional offset
and frequency. The module also includes various helper functions to preprocess the data, ensuring
accurate and efficient rolling sum calculations.

The rolling sum calculation can be tailored to specific needs by adjusting parameters such as the
calculation frequency (`every`), the look-back period (`period`), and the offset from the current
date (`offset`). Furthermore, it supports grouping by one or more categorical variables, enabling
segmented rolling sum calculations within the data.

Example Usage:
--------------
Assuming you have a Polars LazyFrame `lf` with a date column named 'date', a numeric column named 'value',
and optionally, a categorical column named 'category':

```python
import polars as pl

# Sample data creation
data = {
    'date': pl.date_range(start="2020-01-01", end="2020-01-10", interval="1d"),
    'value': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'category': ['A', 'A', 'B', 'B', 'C', 'C', 'A', 'A', 'B', 'B']
}
lf = pl.DataFrame(data).lazy()

# Calculating rolling sum without category columns
rolling_sum = dynamic_rolling_sum(
    lf=lf,
    x='value',
    date_col='date',
    every="1d",
    period="7d",
    offset="0d"
)

# Calculating rolling sum with category columns
rolling_sum_with_categories = dynamic_rolling_sum(
    lf=lf,
    x='value',
    date_col='date',
    category_cols='category',
    every="1d",
    period="7d",
    offset="0d"
)

# To view the result, collect the LazyFrame
print(rolling_sum.collect())
print(rolling_sum_with_categories.collect())

This script aims to provide a flexible and powerful tool for time series analysis, particularly useful for financial analysis, sales forecasting, and any scenario where understanding the cumulative effect over time is crucial.

Notes:
- The date_col should be of datetime type.
- The category_cols parameter can be a single column name or a list of names for multiple categories.
- The every, period, and offset parameters allow for detailed control over the rolling window’s behavior, accommodating various analytical needs.
"""

import polars as pl
from typing import List, Optional, Dict, Union
from datetime import datetime


def dynamic_rolling_sum(
    lf: pl.LazyFrame,
    x: str,
    date_col: str,
    category_cols: Optional[Union[str, List[str]]] = None,
    x_name: Optional[str] = None,
    every: str = "1d",
    period: str = "1y",
    offset: str = "-1mo",
) -> pl.Series:
    """
    Calculate the rolling sum for each row in the lazyframe.

    Parameters
    ----------
    lf : pl.LazyFrame
        The lazyframe to calculate the rolling sum for.
    x : str
        The name of the column to calculate the rolling sum for.
    date_col : str
        The name of the date column.
    category_cols : Union[str, List[str]], optional
        The name of the category column, or a list of names of category columns.
    x_name : str, optional
        The name of the new column to add to the lazyframe.
    every : str, optional
        The frequency of the rolling sum. Default is "1d".
    period : str, optional
        The period of the rolling sum. Default is "1y".
    offset : str, optional
        The offset of the rolling sum. Default is "-1mo".

    Returns
    -------
    pl.Series
        The rolling sum for each row in the lazyframe.
    """
    # Format the date and, optionally, category columns to ensure they
    # are in the correct format
    date_expr = _formatted_date_col(date_col)
    category_cols_ = (
        _formatted_category_cols(category_cols) if category_cols is not None else None
    )
    lf = lf.with_columns(
        [date_expr] + (category_cols_ if category_cols_ is not None else [])
    )

    # Get the lazyframe with the original row order preserved
    lf_order = _get_original_order(lf, date_col, _handle_cat_input(category_cols))

    # Get the name of the new column to add to the lazyframe
    x_name = _get_x_name(x, x_name)

    # Calculate the rolling sum
    if category_cols is not None:
        sum_lf = _rolling_sum_categories(
            lf,
            x,
            date_col,
            _handle_cat_input(category_cols),
            x_name,
            every,
            period,
            offset,
        )
    else:
        sum_lf = _rolling_sum_no_categories(
            lf, x, date_col, x_name, every, period, offset
        )

    # Resort the lazyframe to the original order
    resorted_lf = sum_lf.join(lf_order, on="index").sort("index")

    return resorted_lf.select(pl.col(x_name)).collect().to_series()


def _handle_cat_input(category_cols: Optional[Union[str, List[str]]]) -> List[str]:
    """
    This function ensures that the category_cols parameter is correctly formatted as a list of strings.
    If the input is a single string, it is converted to a list with one element. If the input is already
    a list, it is returned as is. If the input is None, it is returned as None.

    Parameters
    ----------
    category_cols : Optional[Union[str, List[str]]]
        The name of the category column, or a list of names of category columns.

    Returns
    -------
    Optional[List[str]]
        The list of names of the category columns, or None if the input was None.

    Examples
    --------
    >>> _handle_string_or_list_input('category')
    ['category']

    >>> _handle_string_or_list_input(['category1', 'category2'])
    ['category1', 'category2']

    >>> _handle_string_or_list_input(None)
    None
    """
    if category_cols is None:
        return []
    return [category_cols] if isinstance(category_cols, str) else category_cols


def _get_x_name(x: str, x_name: Optional[str]) -> str:
    """
    Determine the appropriate name for the rolling sum column based on provided parameters.
    If a custom name (`x_name`) is not provided, a default name is generated by appending
    '_rolling_sum' to the original column name (`x`).

    Parameters
    ----------
    x : str
        The base name of the column for which the rolling sum is being calculated.
        This name is used to generate the default column name if `x_name` is not provided.

    x_name : Optional[str]
        An optional custom name for the rolling sum column. If specified, this name is used
        directly. If None, a default name based on `x` is generated.

    Returns
    -------
    str
        The determined name for the new rolling sum column. This will be either the custom name
        provided via `x_name` or a default name generated from the base column name `x`.

    Examples
    --------
    >>> _get_x_name('sales', None)
    'sales_rolling_sum'

    >>> _get_x_name('sales', 'monthly_sales_sum')
    'monthly_sales_sum'
    
    Note
    ----
    This function is intended for internal use within the module to ensure consistent naming
    of the newly added rolling sum column in the lazyframe.
    """
    return f"{x}_rolling_sum" if not x_name else x_name

def _get_date_map(lf: pl.LazyFrame, date_col: str) -> Dict[datetime, datetime]:
    """
    Generate a dictionary mapping each date in the specified column of a LazyFrame to its
    "reversed" counterpart. The "reversed" date is defined as the date's position inverted
    within the range from the earliest date in the column to the current date, such that
    the earliest date maps to the current date, and vice versa.

    This mapping is used to facilitate time-based calculations where the ordering of dates
    needs to be temporarily inverted, without altering the original date values.

    Parameters
    ----------
    lf : pl.LazyFrame
        The LazyFrame containing the date data from which to generate the mapping. It is
        assumed that this LazyFrame has at least one row; otherwise, the function's behavior
        is undefined.

    date_col : str
        The name of the column in `lf` that contains date values. This column should be of
        a date-compatible type (e.g., datetime).

    Returns
    -------
    Dict[datetime, datetime]
        A dictionary where each key is a date found in the specified column of the input
        LazyFrame, and each value is the corresponding "reversed" date within the defined
        date range.

    Examples
    --------
    Assuming `lf` contains a date column 'date_col' with dates ranging from 2021-01-01 to
    2021-01-05, the function might return a mapping like:
    
    {
        datetime(2021, 1, 1): datetime(2021, 1, 5),
        datetime(2021, 1, 2): datetime(2021, 1, 4),
        ...
    }

    Note
    ----
    The function calculates the "reversed" date based on the range between the minimum date
    in the specified column and today's date. It is designed for internal use to support
    specific time-based calculations within the module.
    """
    min_date = lf.select(pl.col(date_col).min()).collect().to_series()[0]
    max_date = datetime.today()
    date_range = pl.datetime_range(start=min_date, end=max_date).cast(pl.Date)
    date_df = pl.select(date_range.alias(date_col)).with_columns(
        [
            pl.col(date_col).cast(pl.Date).name.keep(),
            pl.col(date_col).cast(pl.Date).reverse().alias(f"{date_col}_reversed"),
        ]
    )
    date_map = {
        d20: d10 for d10, d20 in zip(date_df[date_col], date_df[f"{date_col}_reversed"])
    }

    return date_map


def _get_original_order(
    lf: pl.LazyFrame, date_col: str, category_cols: Optional[List[str]] = None
) -> pl.LazyFrame:
    """
    This function preserves the original row order of a Polars LazyFrame (`lf`) after various transformations
    or operations that may alter the row order. It achieves this by assigning a unique row index to each
    row before any processing and using this index to restore the original order. The methodology involves
    creating a mapping of dates to their "reversed" counterparts (`date_map`) based on the specified date column.
    This reversed date mapping is temporarily used to sort the data, allowing for operations that depend on
    date ordering (such as rolling calculations). After these operations, the data can be reordered back to
    its original sequence using the initial row index.
    
    Parameters
    ----------
    lf : pl.LazyFrame
        The Polars LazyFrame on which operations altering row order might be performed.
    date_col : str
        The name of the column containing date values. This column is used for generating the
        `date_map` and indirectly influences the process of preserving the original order.
    category_cols : List[str], optional
        Optionally, a list of category column names that might also be used in conjunction
        with the date column for sorting or grouping operations.
    
    Returns
    -------
    pl.LazyFrame
        A LazyFrame with an additional column named 'index' representing the original row
        order. This allows for the restoration of the initial order after processing.
    
    Notes
    -----
    - The function utilizes a unique approach by employing a 'reversed' date column derived from
      the `date_map`. This reversed date is not directly related to the chronological inversion
      but is a technique to maintain consistency in data processing workflows.
    - The restoration of the original order is particularly useful in scenarios where the sequence
      of data points carries significance, such as time series analysis or when operations like
      rolling sums are calculated with respect to time windows.
    - The use of a row index to preserve order is a common technique in data processing, ensuring
      that irrespective of the transformations applied, the original data sequence can be retrieved.
    
    Example Usage
    -------------
    Assuming `lf` is a LazyFrame with a date column 'date' and an optional category column 'category':
    
    ```python
    # Original LazyFrame creation
    lf = pl.DataFrame({
        'date': pl.date_range(start="2022-01-01", end="2022-01-10", interval="1d"),
        'category': ['A', 'A', 'B', 'B', 'A', 'A', 'B', 'B', 'A', 'A'],
        'value': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    }).lazy()
    
    # Applying transformations that alter row order
    transformed_lf = lf.sort('value', reverse=True)
    
    # Restoring original order using '_get_original_order'
    restored_lf = _get_original_order(transformed_lf, 'date', ['category'])

    # 'restored_lf' now has its rows ordered as in the original 'lf', ready for further analysis
    """
    date_map = _get_date_map(lf, date_col)
    return (
        lf.with_row_index()
        .with_columns(
            [
                # Add the reversed date column
                pl.col(date_col)
                .replace(date_map)
                .cast(pl.Date)
                .alias(f"{date_col}_reversed")
            ]
        )
        .sort([f"{date_col}_reversed"] + _handle_cat_input(category_cols))
        .select(pl.col("index"))
    )


def _formatted_category_cols(category_cols: Union[str, List[str]]) -> List[pl.Expr]:
    """
    This function ensures that category columns in a Polars LazyFrame are properly formatted for analysis.
    It takes a single column name or a list of column names, treating each as a category column. The function
    then casts each specified column to UTF8 (string) format before converting it to a Categorical type. This
    formatting is crucial for efficient memory usage and faster operations on category data in Polars, especially
    useful in grouping, sorting, and other operations where categorical distinctions are necessary.

    Parameters
    ----------
    category_cols : Union[str, List[str]]
        The name of the category column, or a list of names of category columns to be formatted. If a single
        string is provided, it is treated as a list with one element.

    Returns
    -------
    List[pl.Expr]
        A list of Polars expressions for the formatted category columns. Each expression corresponds to a column
        in the input `category_cols`, cast to UTF8 and then to Categorical type.

    Notes
    -----
    - Casting to UTF8 before Categorical is essential because Polars requires a string type before converting
      to a categorical type, ensuring compatibility and optimization for categorical operations.
    - The use of the Categorical type can significantly improve performance in operations that rely on category
      distinctions, such as groupings or pivot tables, due to optimized memory usage and faster computations.
    - This function is particularly useful in the preprocessing steps of data analysis workflows where categorical
      data needs to be standardized across multiple columns.

    Example Usage
    -------------
    Assuming `lf` is a Polars LazyFrame with the columns 'category1' and 'category2' that you wish to format:

    ```python
    # Original LazyFrame creation
    lf = pl.DataFrame({
        'category1': ['A', 'B', 'C', 'A', 'B'],
        'category2': ['X', 'Y', 'X', 'Y', 'Z'],
        'value': [1, 2, 3, 4, 5]
    }).lazy()

    # Formatting the category columns
    formatted_cols = _formatted_category_cols(['category1', 'category2'])

    # Applying the formatted expressions to the LazyFrame
    lf = lf.with_columns(formatted_cols)

    # 'lf' now contains 'category1' and 'category2' as formatted categorical columns,
    # optimized for subsequent categorical operations.
    ```
    """
    return [
        pl.col(col).cast(pl.Utf8).cast(pl.Categorical).name.keep()
        for col in _handle_cat_input(category_cols)
    ]


def _formatted_date_col(date_col: Union[str, pl.Expr]) -> pl.Expr:
    """
    Format the date column to ensure it is in the correct format.

    Parameters
    ----------
    date_col : Union[str, pl.Expr]
        The name of the date column, or an expression for a date column.

    Returns
    -------
    pl.Expr
        An expression for the formatted date column.

    """
    expr = pl.col(date_col) if isinstance(date_col, str) else date_col
    return expr.cast(pl.Date)


def _reversed_date_col(lf: pl.LazyFrame, date_col: str) -> pl.Expr:
    """
    Add a column for the reversed date.

    Parameters
    ----------
    lf : pl.LazyFrame
        The lazyframe containing the date column.
    date_col : str
        The name of the date column.

    Returns
    -------
    pl.Expr
        The reversed date column.
    """
    date_map = _get_date_map(lf, date_col)
    reversed_expr = pl.col(date_col).replace(date_map)
    return _formatted_date_col(reversed_expr)


def _group_by_no_categories(
    lf: pl.LazyFrame,
    reversed_date_expr: pl.Expr,
    x: str,
    x_name: Optional[str],
    every: str = "1d",
    period: str = "1y",
    offset: str = "-1mo",
) -> pl.LazyFrame.group_by.LazyGroupBy:
    """
    Group the lazyframe by the reversed date column. This function is used when there
    are no category columns to group by.

    Parameters
    ----------
    lf : pl.LazyFrame
        The lazyframe to group by.
    reversed_date_expr : pl.Expr
        The expression for the reversed date column.
    x : str
        The name of the column to calculate the rolling sum for.
    x_name : str, optional
        The name of the new column to add to the lazyframe.
    every : str, optional
        The frequency of the rolling sum. Default is "1d".
    period : str, optional
        The period of the rolling sum. Default is "1y".
    offset : str, optional
        The offset of the rolling sum. Default is "-1mo".

    Returns
    -------
    pl.LazyFrame.group_by.LazyGroupBy
        The grouped lazyframe.
    """
    return lf.sort(reversed_date_expr).group_by_dynamic(
        reversed_date_expr,
        every=every,
        period=period,
        offset=offset,
        start_by="datapoint",
        check_sorted=False,
    )


def _group_by_categories(
    lf: pl.LazyFrame,
    reversed_date_expr: pl.Expr,
    x: str,
    x_name: Optional[str],
    category_cols: List[str],
    every: str = "1d",
    period: str = "1y",
    offset: str = "-1mo",
) -> pl.LazyFrame.group_by.LazyGroupBy:
    """
    Group the lazyframe by the reversed date column and the category columns.
    This function is used when there are category columns to group by.

    Parameters
    ----------
    lf : pl.LazyFrame
        The lazyframe to group by.
    reversed_date_expr : pl.Expr
        The expression for the reversed date column.
    x : str
        The name of the column to calculate the rolling sum for.
    x_name : str, optional
        The name of the new column to add to the lazyframe.
    category_cols : List[str]
        The list of names of the categorical columns.
    every : str, optional
        The frequency of the rolling sum. Default is "1d".
    period : str, optional
        The period of the rolling sum. Default is "1y".
    offset : str, optional
        The offset of the rolling sum. Default is "-1mo".

    Returns
    -------
    pl.LazyFrame.group_by.LazyGroupBy
        The grouped lazyframe.
    """
    return (
        lf.with_columns(_formatted_category_cols(category_cols))
        .sort(reversed_date_expr, *category_cols)
        .group_by_dynamic(
            reversed_date_expr,
            every=every,
            period=period,
            offset=offset,
            by=category_cols,
            start_by="datapoint",
            check_sorted=False,
        )
    )


def _rolling_sum_no_categories(
    lf: pl.LazyFrame,
    x: str,
    date_col: str,
    x_name: Optional[str],
    every: str = "1d",
    period: str = "1y",
    offset: str = "-1mo",
) -> pl.LazyFrame:
    """
    Calculate the rolling sum for each row in the lazyframe. This function is used when there are no category columns to group by.

    Parameters
    ----------
    lf : pl.LazyFrame
        The lazyframe to calculate the rolling sum for.
    x : str
        The name of the column to calculate the rolling sum for.
    date_col : str
        The name of the date column.
    x_name : str, optional
        The name of the new column to add to the lazyframe.
    every : str, optional
        The frequency of the rolling sum. Default is "1d".
    period : str, optional
        The period of the rolling sum. Default is "1y".
    offset : str, optional
        The offset of the rolling sum. Default is "-1mo".

    Returns
    -------
    pl.LazyFrame
        The lazyframe with the rolling sum added as a new column.
    """
    # Get the lazyframe with the original row order preserved
    lf_order = _get_original_order(lf, date_col)

    # Add a column for the reversed date
    reversed_date_expr = _reversed_date_col(lf, date_col)

    # Use the reverse-sorted date to calculate the rolling sum
    lfgby = _group_by_no_categories(
        lf_order, reversed_date_expr, x, x_name, every, period, offset
    )

    # Calculate the rolling sum
    return lfgby.agg(pl.sum(x).name.keep())


def _rolling_sum_categories(
    lf: pl.LazyFrame,
    x: str,
    date_col: str,
    category_cols: List[str],
    x_name: Optional[str],
    every: str = "1d",
    period: str = "1y",
    offset: str = "-1mo",
) -> pl.LazyFrame:
    """
    Calculate the rolling sum for each row in the lazyframe. This function is used when there are category columns to group by.

    Parameters
    ----------
    lf : pl.LazyFrame
        The lazyframe to calculate the rolling sum for.
    x : str
        The name of the column to calculate the rolling sum for.
    date_col : str
        The name of the date column.
    category_cols : List[str]
        The list of names of the categorical columns.
    x_name : str, optional
        The name of the new column to add to the lazyframe.
    every : str, optional
        The frequency of the rolling sum. Default is "1d".
    period : str, optional
        The period of the rolling sum. Default is "1y".
    offset : str, optional
        The offset of the rolling sum. Default is "-1mo".

    Returns
    -------
    pl.LazyFrame
        The lazyframe with the rolling sum added as a new column.
    """
    # Get the lazyframe with the original row order preserved
    lf_order = _get_original_order(lf, date_col, category_cols)

    # Add a column for the reversed date
    reversed_date_expr = _reversed_date_col(lf, date_col)

    # Use the reverse-sorted date to calculate the rolling sum
    lfgby = _group_by_categories(
        lf_order, reversed_date_expr, x, x_name, category_cols, every, period, offset
    )

    # Calculate the rolling sum
    return lfgby.agg(pl.sum(x).name.keep())
