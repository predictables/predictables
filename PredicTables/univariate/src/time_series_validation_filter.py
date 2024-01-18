from typing import Union

import pandas as pd
import polars as pl

from PredicTables.util import to_pl_lf


def time_series_validation_filter(
    df: Union[pd.DataFrame, pl.DataFrame, pl.LazyFrame],
    df_val: Union[pd.DataFrame, pl.DataFrame, pl.LazyFrame] = None,
    fold: int = None,
    fold_col: str = "cv",
    feature_col: str = None,
    target_col: str = None,
    time_series_validation: bool = False,
):
    df = to_pl_lf(df)

    # allowed to pass in a separate validation set, but if none passed (or if doing cross
    # validation (and so validation set is part of the training set)), use the training set
    if df_val is None:
        df_val_start = df
    else:
        if fold is None:
            # If no fold passed, assume not doing a cross-validation split here
            df_val_start = to_pl_lf(df_val)
        else:
            # If a fold is passed, we are doing cross validation so don't use any
            # validation set
            df_val_start = None

    if fold is None:
        # If no fold passed, assume not doing a cross-validation split here
        df_train = df.select([feature_col, target_col])
        df_test = df_val_start.select([feature_col, target_col])
    else:
        # If a fold is passed, we are doing cross validation so don't use any
        # validation set
        df = df.select([feature_col, target_col, fold_col])
        if time_series_validation:
            df_train = df.filter(pl.col(fold_col) < fold)
            df_test = df.filter(pl.col(fold_col) >= fold)
        else:
            df_train = df.filter(pl.col(fold_col) != fold)
            df_test = df.filter(pl.col(fold_col) == fold)

    # Split into X, y and train, test
    X_train = df_train.select([feature_col]).collect().to_pandas()
    y_train = df_train.select([target_col]).collect().to_pandas()[target_col]
    X_test = df_test.select([feature_col]).collect().to_pandas()
    y_test = df_test.select([target_col]).collect().to_pandas()[target_col]

    return X_train, y_train, X_test, y_test