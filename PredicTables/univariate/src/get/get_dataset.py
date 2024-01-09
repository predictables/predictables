from typing import Callable, Optional
import pandas as pd
import polars as pl

from PredicTables.util import get_column_dtype


def get_data(
    folder: str = "/sas/data/project/EG/ActShared/SmallBusiness/Modeling/Hit Ratio/bop_model/",
    filename: str = "X_train_ts_raw.parquet",
    cv_fold_filename: str = "cv_folds.parquet",
    cv_fold: int = None,
    feature: list | str = None,
    target: str = None,
    denom: str = None,
    na_rm: bool = False,
    standardize: bool = True,
    return_std_params: bool = False,
) -> pl.LazyFrame:
    """
    Returns the training dataset. Optionally filters by fold number and/or
    removes rows with missing values.
    """
    # Get data with target, feature, and denom columns
    if isinstance(feature, str):
        feature = [feature]
    df = pl.scan_parquet(folder + filename)[feature + [target, denom]]

    # Get fold column (if cv_fold is not None)
    if cv_fold is not None:
        df = df.join(pl.scan_parquet(folder + cv_fold_filename)["fold"])

    if na_rm:
        df = df.drop_nulls()

    if feature is not None:
        feature_dtype = get_column_dtype(df, feature)
        # if the feature is categorical, convert to dummy variables
        if feature_dtype == "categorical":
            df = df.join(df.select(feature).collect().to_dummies())

    if target is not None:
        target_dtype = get_column_dtype(df, target)
        # if the target is binary, convert to 0/1
        if target_dtype == "binary":
            df = df.with_columns([pl.col(target).cast(pl.Float32).alias(target)])
            model_type = "logistic"
        # if the target is continuous, convert to float
        elif target_dtype == "continuous":
            df = df.with_columns([pl.col(target).cast(pl.Float32).alias(target)])
            model_type = "linear"
        # if the target is categorical, convert to dummy variables
        elif target_dtype == "categorical":
            df = df.join(df.select(target).collect().to_dummies())
            model_type = "logistic"

        else:
            raise ValueError(
                f"Target column {target} has dtype {target_dtype}, which is not supported."
            )

    if denom is not None:
        denom_dtype = get_column_dtype(df, denom)
        # if the denom is binary, convert to 0/1
        if denom_dtype == "binary":
            df = df.with_columns([pl.col(denom).cast(pl.Float32).alias(denom)])
        # if the denom is continuous, convert to float
        elif denom_dtype == "continuous":
            df = df.with_columns([pl.col(denom).cast(pl.Float32).alias(denom)])
        # if the denom is categorical, convert to dummy variables
        elif denom_dtype == "categorical":
            df = df.join(df.select(denom).collect().to_dummies())
        else:
            raise ValueError(
                f"Denominator column {denom} has dtype {denom_dtype}, which is not supported."
            )

        # get max target for each lowest grain to ensure you get a hit if there
        # is at least one
        max_target = (
            df.groupby("lowest_grain")[self.target]
            .max()
            .reset_index()
            .drop_duplicates()
        )

        # join back to the df to get the feature and denom corresponding to the
        # max target
        df = df.merge(max_target, on="lowest_grain", how="inner")

        # drop the max target column
        df.drop(columns=self.target + "_y", inplace=True)

        # rename the max target column to target
        df.rename(columns={self.target + "_x": self.target}, inplace=True)

        # drop duplicates
        df.drop_duplicates(inplace=True)

    if standardize:
        if self.type == "continuous":
            mu = df[self.feature].mean()
            sigma = df[self.feature].std()
            df[self.feature] = (df[self.feature] - mu) / sigma

    if return_std_params:
        return df, mu, sigma
    else:
        return df
