from typing import Optional, Union

import pandas as pd  # type: ignore
import polars as pl

from predictables.util import cv_filter, to_pl_lf


def time_series_validation_filter(
    df: Union[pd.DataFrame, pl.DataFrame, pl.LazyFrame],
    df_val: Union[pd.DataFrame, pl.DataFrame, pl.LazyFrame],
    fold: Optional[int] = None,
    fold_col: str = "cv",
    feature_col: str = "feature",
    target_col: str = "target",
    time_series_validation: bool = False,
) -> tuple:
    """
    Filter a dataframe into train and test sets for time series validation.
    """
    df = to_pl_lf(df)

    # allowed to pass in a separate validation set, but if none passed (or if
    # doing cross validation (and so validation set is part of the training
    # set)), use the training set
    df_val_start = (
        df if df_val is None else (to_pl_lf(df_val) if fold is None else None)
    )

    if fold is None:
        # If no fold passed, assume not doing a cross-validation split here
        df_train = df.select([feature_col, target_col])
        df_test = (
            df_val_start.select([feature_col, target_col])
            if df_val_start is not None
            else None
        )
    else:
        # If a fold is passed, we are doing cross validation so don't use any
        # validation set
        df = df.select(
            [
                pl.col(feature_col).name.keep(),
                pl.col(target_col).name.keep(),
                pl.col(fold_col).name.keep(),
            ]
        ).with_row_index()

        def train() -> pl.Series:
            x = pl.from_pandas(
                cv_filter(
                    fold,
                    df.select(fold_col).collect()[fold_col],
                    time_series_validation=time_series_validation,
                    train_test="train",
                )
            )
            return x if isinstance()
            .to_series()

        def test() -> pl.Series:
            return pl.from_pandas(
                cv_filter(
                    fold,
                    df.select(fold_col).collect()[fold_col],
                    time_series_validation=time_series_validation,
                    train_test="test",
                )
            ).to_series()

        df_train = df.filter(train())
        df_test = df.filter(test())

    # Split into X, y and train, test
    X_test = (
        df_test.collect().to_pandas()[[feature_col]] if df_test is not None else None
    )
    y_test = (
        df_test.select([target_col]).collect().to_pandas()[target_col]
        if df_test is not None
        else None
    )

    X_train = df_train.collect().to_pandas()[[feature_col]]
    y_train = df_train.select([target_col]).collect().to_pandas()[target_col]

    return X_train, y_train, X_test, y_test
