# import polars as pl
# from predictables.encoding.src.lagged_mean_encoding._dynamic_rolling_sum import (
#     dynamic_rolling_sum,
# )
# from predictables.util import validate_lf
# from typing import List, Any, Union


def dynamic_rolling_sum_with_categories():
    print("h1")


# @validate_lf
# def dynamic_rolling_sum_with_categories(
#     lf: pl.LazyFrame,
#     x_col: str,
#     date_col: str,
#     category_cols: Union[str, List[str]],
#     index_col: str = "index",
#     offset: int = 30,
#     window: int = 360,
# ) -> pl.LazyFrame:

#     # Check to ensure there is an index column:
#     if index_col not in lf.columns:
#         raise ValueError(
#             f"Index column {index_col} not found in LazyFrame. "
#             "Please provide a valid index column."
#         )

#     if isinstance(category_cols, str):
#         category_cols = [category_cols]

#     # Data frame with the row index as a column, to resort the data
#     # to the original order after the rolling sum
#     lf_order = lf.select([pl.col(index_col).alias("index")])

#     # # Combine category cols into a single struct column
#     # lf = _combine_categories_into_struct(lf, category_cols)

#     # Get unique combinations of category columns
#     unique_categories = _get_unique_categories(lf, category_cols)

#     # Create empty list to store the results
#     lf_results_list = []

#     # For each unique category combination
#     for category in unique_categories:
#         # Filter the LazyFrame to only include rows with the current
#         # category combination

#         lf_category = _filter_category(lf, category)

#         # Perform the dynamic rolling sum
#         lf_category = dynamic_rolling_sum(
#             lf_category, x_col, date_col, index_col, offset, window
#         )

#         # Append the results to the results list
#         lf_results_list.append(lf_category)

#     # Combine the results into a single LazyFrame
#     lf_results = pl.concat(
#         lf_results_list,
#         how="vertical",
#     )

#     # Resort the data to the original order
#     return lf_order.join(lf_results, on="index", how="left")


# def _combine_categories_into_struct(
#     lf: pl.LazyFrame, category_cols: List[str]
# ) -> pl.LazyFrame:
#     """
#     Takes a LazyFrame and a list of category columns, and returns a LazyFrame
#     with the category columns combined into a single struct column.
#     """
#     return lf.with_columns([pl.struct(category_cols).alias("category")])


# def _get_unique_categories(
#     lf: pl.LazyFrame, category_cols: List[str]
# ) -> List[pl.Struct]:
#     """
#     Takes a LazyFrame and the name of the struct column, and returns a list of the
#     unique combinations of the category columns.
#     """
#     return lf.select(category_cols).unique().collect().to_series().to_list()


# def _filter_category(lf: pl.LazyFrame, col:str, category:
#     """
#     Takes a LazyFrame, a category, and the name of the struct column, and returns a
#     LazyFrame with only the rows that match the category.
#     """
#     breakpoint()
#     for cat in category_cols:
#         lf = lf.filter(pl.col("category").get(cat) == cat)

#     return lf
