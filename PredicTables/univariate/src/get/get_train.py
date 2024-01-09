from typing import Callable, Optional
import pandas as pd
import polars as pl


def GetTrain(
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
) -> pd.DataFrame:
    """
    Returns the training dataset. Optionally filters by fold number and/or
    removes rows with missing values.
    """
    df = self.train.copy()[[self.target, self.feature, self.denom]]
    df["fold"] = self.fold
    df["feature"] = df[self.feature]
    df["target"] = df[self.target]
    df["denom"] = df[self.denom]
    if self.type == "categorical":
        df = pd.concat([df, self.GetDummies("train")], axis=1)
    if fold is not None:
        if isinstance(fold, int):
            df = df.loc[df["fold"] == fold]
        elif callable(fold):
            df = df.loc[df["fold"].apply(fold)]

    if na_rm:
        df.dropna(inplace=True)

    if feature is not None:
        if isinstance(feature, str):
            df = df.loc[df[self.feature] == feature]
        elif isinstance(feature, float):
            df = df.loc[df[self.feature] == feature]
        elif isinstance(feature, int):
            df = df.loc[df[self.feature] == feature]
        elif callable(feature):
            df = df.loc[df[self.feature].apply(feature)]

    if target is not None:
        if isinstance(target, str):
            df = df.loc[df[self.target] == target]
        elif isinstance(target, float):
            df = df.loc[df[self.target] == target]
        elif isinstance(target, int):
            df = df.loc[df[self.target] == target]
        elif callable(target):
            df = df.loc[df[self.target].apply(target)]

    if denom is not None:
        if isinstance(denom, str):
            df = df.loc[df[self.denom] == denom]
        elif isinstance(denom, float):
            df = df.loc[df[self.denom] == denom]
        elif isinstance(denom, int):
            df = df.loc[df[self.denom] == denom]
        elif callable(denom):
            df = df.loc[df[self.denom].apply(denom)]

    if unique_lowest_grain:
        df["lowest_grain"] = self.df.loc[
            self.df.index.to_series().isin(self.train.index.tolist())
        ][self.lowest_grain].values

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
