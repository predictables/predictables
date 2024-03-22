"""Notes on the implementation of the DynamicRollingSum class.

1. validate parameters
2. If categories passed:
    a. ensure category_col is a list (if a string, convert to list)
    b. loop through each category_col
        i. get unique levels
        ii. create a column name for the rolling sum
        iii. loop through each unique level
            1. filter the LazyFrame to only include rows where the category_col is equal to the level
            2. run the dynamic rolling sum on the filtered LazyFrame
            3. add the level as a column to the resulting LazyFrame
        iv. append the lf's from each level to a list and concatenate them vertically
        v. cast the category_col to a string
        vi. join the original LazyFrame with the concatenated LazyFrame
        vii. recast the category_col to a categorical
    c. drop any columns with "_right" as a suffix
    d. return a lazyframe with:
        i. index
        ii. date
        iii. the category_col(s)
        iv. the rolling sum
3. If no categories passed:
    a. create a column name for the rolling sum
    b. run the dynamic rolling sum on the LazyFrame
4. Take output from either 2 or 3
5. If no category columns are passed:
    a. get the column name for the rolling sum
    b. if rejoin is True:
        - we are adding the rolling sum to the original LazyFrame
        i. join the rolling sum to the original LazyFrame on the index column
        ii. return the LazyFrame
    c. if rejoin is False:
        - we are returning a lf with columns for the index, date, and rolling sum only
        i. return the LazyFrame unchanged
6. If category columns are passed:
    a. get the column name for the rolling sum
    b. if rejoin is True:
        - we are adding the rolling sum to the original LazyFrame
        i. select:
            1. index
            2. rolling sum columns
        ii. cast each category column in the original LazyFrame to a string
        iii. join the rolling sum to the original LazyFrame on the index column
        iv. recast each category column to a categorical
        v. return the LazyFrame
    c. if rejoin is False:
        - we are returning a lf with columns for the index, date, categories, and rolling sum
        i. select:
            1. index
            2. date
            3. category columns
            4. rolling sum columns
        ii. cast each category column in the original LazyFrame to a string
        iii. return the LazyFrame

7. Before returning the LazyFrame, drop any columns with "_right" as a suffix


"""
# def _run(self) -> pl.LazyFrame:


#         # Check to see whether a x_name has been set

#         if self._category_cols is not None:
#             # if isinstance(self._category_cols, str):
#             #     self._category_cols = [self._category_cols]

#             for c in self._category_cols:
# unique_levels = (
#     self._lf.select(c).unique().collect().to_series().to_list()
# )
# col_name = rolling_op_column_name(
#     self._op, self._x_name, c, self._offset, self._window
# )
# cat_dfs = [
#     dynamic_rolling_sum(
#         lf=self._lf.filter(pl.col(c).cast(pl.Utf8) == str(level)),
#         x_col=self._x_col,
#         date_col=self._date_col,
#         index_col=self._index_col,
#         offset=self._offset,
#         window=self._window,
#     )
#     .with_columns([pl.lit(str(level)).alias(c)])
#     .with_columns(pl.col("rolling_value_list").alias(col_name))
#     .drop(["rolling_value_list"])
#     .sort(["index", c])
#     for level in unique_levels
# ]

# self._lf = (
#     self._lf
#     .with_columns([pl.col(c).cast(pl.Utf8).name.keep()])
#     .join(
#         pl.concat(cat_dfs, how="vertical"), on=["index", c], how="left"
#     )
#     .with_columns([pl.col(c).cast(pl.Categorical).name.keep()])
# )

# if "date_right" in self._lf.columns:
#     self._lf = self._lf.drop(["date_right"])

# return self._lf.select(
#     [
#         pl.col(self._index_col).name.keep(),
#         pl.col(self._date_col).name.keep(),
#     ]
#     + [
#         pl.col(c).cast(pl.Utf8).cast(pl.Categorical).name.keep()
#         for c in self._category_cols
#     ]
#     + [pl.col(f"{col_name}").name.keep()]
# )
# else:
# Run the dynamic rolling sum if all parameters are set
# col_name = rolling_op_column_name(
#     self._op, self._x_name, None, self._offset, self._window
# )
#             return (
#                 dynamic_rolling_sum(
#                     lf=self._lf,
#                     x_col=self._x_col,
#                     date_col=self._date_col,
#                     index_col=self._index_col,
#                     offset=self._offset,
#                     window=self._window,
#                 )
#                 .with_columns([pl.col("rolling_value_list").alias(col_name)])
#                 .drop("rolling_value_list")
#             )


# def run(self) -> pl.LazyFrame:
#         """Process this code.

#         1. Get output from the _run method


#         """
#         # Run the dynamic rolling sum if all parameters are set
#         lf = self._run()

#         # If rejoin is True, concatenate the rolling sum to the original LazyFrame
#         # Otherwise, return a LazyFrame with the same number of rows, but with columns
#         # for the index, date, categories, and rolling sum
#         if self._category_cols is None:
#             # col = self._get_column_name()
#             # out = (
#             #     self._lf.join(
#             #         lf.select(pl.col(col)), on=[self._index_col], how="left"
#             #     )
#             #     if self._rejoin
#             #     else lf
#             # )

#         else:
#             # lf = lf.select(
#             #     [pl.col(self._index_col).name.keep()]
#             #     + [
#             #         rolling_op_column_name(
#             #             self._op, self._x_name, c, self._offset, self._window
#             #         )
#             #         for c in self._category_cols
#             #     ]
#             # )

#             out = (
#                 self._lf.with_columns(
#                     [pl.col(c).cast(pl.Utf8).name.keep() for c in self._category_cols]
#                 )
#                 .join(
#                     lf.select(
#                         [
#                             pl.col(c).name.keep()
#                             for c in lf.columns
#                             if c != self._date_col
#                         ]
#                     ),
#                     on=[self._index_col],
#                     how="left",
#                 )
#                 .with_columns(
#                     [
#                         pl.col(c).cast(pl.Categorical).name.keep()
#                         for c in self._category_cols
#                     ]
#                 )
#                 if self._rejoin

#                 else lf
#             )

#         # If there are any columns suffixed with "_right", drop them
#         if any([c.endswith("_right") for c in out.columns]):
#             out = out.drop([c for c in out.columns if c.endswith("_right")])

#         if any([c.lower().strip() == "index" for c in out.columns]):
#             out = out.drop([c for c in out.columns if c.lower().strip() == "index"])

#         return out
