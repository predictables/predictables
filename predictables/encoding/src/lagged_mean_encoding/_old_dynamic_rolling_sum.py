# def calculate_rolling_sum(
#     lf: pl.LazyFrame,
#     x: str,
#     date_col: str,
#     category_cols: Optional[List[str]],
#     x_name: Optional[str],
# ) -> pl.Series:
#     """
#     Calculate the rolling sum for each row in the lazyframe.

#     Parameters
#     ----------
#     lf : pl.LazyFrame
#         The lazyframe to calculate the rolling sum for.
#     x : str
#         The name of the column to calculate the rolling sum for.
#     date_col : str
#         The name of the date column.
#     category_cols : List[str], optional
#         The list of names of the categorical columns.
#     x_name : str, optional
#         The name of the new column to add to the lazyframe.

#     Returns
#     -------
#     pl.Series
#         The rolling sum for each row.
#     """
#     # Get the lazyframe with the original row order preserved
#     lf_order = _get_original_order(lf, date_col, category_cols)

#     # Format the date column to ensure it is in the correct format
#     date_expr = _formatted_date_col(date_col)

#     # Add a column for the reversed date
#     reversed_date_expr = _reversed_date_col(lf, date_col)

#     # Use the reverse-sorted date to calculate the rolling sum
#     if category_cols is not None:
#         # Handle the case where there are category columns
#         lfgby = (
#             lf.with_columns(
#                 [pl.col(col).cast(pl.Categorical).name.keep() for col in category_cols]
#             )
#             .sort(reversed_date_expr, *category_cols)
#             .group_by_dynamic(
#                 reversed_date_expr,
#                 every="1d",
#                 period="1y",
#                 offset="-1mo",
#                 by=category_cols,
#                 start_by="datapoint",
#                 check_sorted=False,
#             )
#         )
#     else:
#         # Handle the case where there are no category columns
#         lfgby = lf.sort(reversed_date_expr).group_by_dynamic(
#             reversed_date_expr,
#             every="1d",
#             period="1y",
#             offset="-1mo",
#             start_by="datapoint",
#             check_sorted=False,
#         )

#     nm = f"{x}_rolling_sum" if x_name is None else x_name

#     # Return the rolling sum of the values in column x
#     return (
#         lfgby.agg(
#             # Add the rolling sum column
#             [pl.col(x).cast(pl.Float64).sum().alias(nm)]
#         )
#         .with_context(
#             # Add the original row index to the lazyframe
#             lf_order
#         )
#         .sort(
#             # Restore the original row order
#             "index"
#         )
#         .select(
#             # Only returning the rolling sum column
#             pl.col(nm)
#             .cast(pl.Float64)
#             .name.keep()
#         )
#         .collect()
#         .to_series()  # Convert to a series
#     )


# The original function for reference
# def dynamic_rolling_sum(
#     lf: pl.LazyFrame,
#     x: str,
#     date_col: str,
#     category_cols: Optional[List[str]] = None,
#     x_name: Optional[str] = None,
# ) -> pl.Series:
#     """
#     For each row in the lf, calculate the sum of the values in column x for the year
#     prior to the date in that row, segmented by the category columns and starting at the
#     date one month prior to the date in that row.

#     Returns a pl.Series, re-sorting the rows to match the original row order. This
#     can then be dropped into the original lazyframe.

#     Parameters
#     ----------
#     lf : pl.LazyFrame
#         The lazyframe.
#     x : str
#         The name of the column to calculate the rolling sum for.
#     date_col : str
#         The name of the date column.
#     category_cols : List[str], optional
#         The list of names of the categorical columns.
#     x_name : str, optional
#         The name of the new column to add to the lazyframe. If None, the new column will
#         be named f"{x}_rolling_sum".

#     Returns
#     -------
#     pl.Series
#         The rolling sum of the values in column x for the year prior to the date
#         in that row. This series will be sorted to match the original row order,
#         and is intended to drop into the original lazyframe.

#     Notes
#     -----
#     The rolling sum is calculated using the following steps:
#         1. Make sure the date is in the correct format, and add a column for the
#         reversed date.
#         2. Use the reverse-sorted date here to calculate the rolling sum for the year
#         PRIOR to the date one month prior to the date in that row.
#         3. Return the rolling sum of the values in column x over that window.

#     """
#     # Get the dictionary mappings for the date column and its
#     # reversed date
#     date_map = _get_date_map(lf, date_col)

#     # Get a lazy frame with the original row order preserved
#     lf_order = (
#         lf.with_row_index()
#         .with_columns(
#             [
#                 # Add the reversed date column
#                 pl.col(date_col)
#                 .replace(date_map)
#                 .cast(pl.Date)
#                 .alias(f"{date_col}_reversed")
#             ]
#         )
#         .sort(
#             [f"{date_col}_reversed"]
#             + (category_cols if category_cols is not None else [])
#         )
#     )

#     # Make sure the date is in the correct format, and
#     # add a column for the reversed date
#     lf = lf.with_columns(
#         [
#             pl.col(date_col).cast(pl.Date).name.keep(),
#             pl.col(date_col)
#             .replace(date_map)
#             .cast(pl.Date)
#             .alias(f"{date_col}_reversed"),
#         ]
#     )

#     # Use the reverse-sorted date here to calculate the rolling sum
#     # for the year PRIOR to the date in that row
#     if category_cols is not None:
#         # Handle the case where there are category columns
#         lfgby = (
#             lf.with_columns(
#                 [pl.col(col).cast(pl.Categorical).name.keep() for col in category_cols]
#             )
#             .sort(f"{date_col}_reversed", *category_cols)
#             .group_by_dynamic(
#                 f"{date_col}_reversed",
#                 every="1d",
#                 period="1y",
#                 offset="-1mo",
#                 by=category_cols,
#                 start_by="datapoint",
#                 check_sorted=False,
#             )
#         )
#     else:
#         # Handle the case where there are no category columns
#         lfgby = lf.sort(f"{date_col}_reversed").group_by_dynamic(
#             f"{date_col}_reversed",
#             every="1d",
#             period="1y",
#             offset="-1mo",
#             start_by="datapoint",
#             check_sorted=False,
#         )

#     nm = f"{x}_rolling_sum" if x_name is None else x_name

#     # Return the rolling sum of the values in column x
#     return (
#         lfgby.agg(
#             # Add the rolling sum column
#             [pl.col(x).cast(pl.Float64).sum().alias(nm)]
#         )
#         .with_context(
#             # Add the original row index to the lazyframe
#             lf_order.select(pl.col("index"))
#         )
#         .sort(
#             # Restore the original row order
#             "index"
#         )
#         .select(
#             # Only returning the rolling sum column
#             pl.col(nm)
#             .cast(pl.Float64)
#             .name.keep()
#         )
#         .collect()
#         .to_series()  # Convert to a series
#     )
