# import pandas as pd
# import polars as pl

# from typing import Union
# from PredicTables.util import to_pl_lf


# def extract_lowest_grain(
#     train_df: Union[pd.DataFrame, pl.DataFrame, pl.LazyFrame],
#     val_df: Union[pd.DataFrame, pl.DataFrame, pl.LazyFrame],
#     test_df: Union[pd.DataFrame, pl.DataFrame, pl.LazyFrame],
#     lowest_grain: str = "evolve_eppid",
# ):
#     """
#     Extracts the lowest grain (defaults to EPPID) from the provided DataFrames.

#     Parameters
#     ----------
#     train_df : Union[pd.DataFrame, pl.DataFrame, pl.LazyFrame]
#         The training DataFrame.
#     val_df : Union[pd.DataFrame, pl.DataFrame, pl.LazyFrame]
#         The validation DataFrame.
#     test_df : Union[pd.DataFrame, pl.DataFrame, pl.LazyFrame]
#         The testing DataFrame.

#     Returns
#     -------
#     Tuple[Union[pd.Series, pl.Series], Union[pd.Series, pl.Series], Union[pd.Series, pl.Series]]
#         The lowest grain for the training, validation, and testing DataFrames.
#     """
#     train_lf = to_pl_lf(train_df)
#     val_lf = to_pl_lf(val_df)
#     test_lf = to_pl_lf(test_df)

#     train_lowest_grain = train_lf.select([lowest_grain]).unique()
#     val_lowest_grain = val_lf.select([lowest_grain]).unique()

#     return train_lowest_grain, val_lowest_grain, test_lowest_grain
