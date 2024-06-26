from __future__ import annotations


import pandas as pd
import polars as pl

from predictables.core.src._UnivariateAnalysis import UnivariateAnalysis
from predictables.util import to_pd_df, to_pd_s


class PredicTables:
    feature_column_names: list[str]

    def __init__(
        self,
        model_name: str,
        df_train: pd.DataFrame | pl.DataFrame | pl.LazyFrame,
        df_val: pd.DataFrame | pl.DataFrame | pl.LazyFrame,
        df_test: pd.DataFrame | pl.DataFrame | pl.LazyFrame,
        target_column_name: str,
        cv_folds: pd.Series | pl.Series,
        has_time_series_structure: bool = False,
    ):
        self.model_name = model_name
        self.cv_folds = to_pd_s(cv_folds)
        self.df_train = to_pd_df(df_train).assign(cv=self.cv_folds)
        self.df_val = to_pd_df(df_val)
        self.df_test = to_pd_df(df_test)

        self.target_column_name = target_column_name
        self.has_time_series_structure = has_time_series_structure

        self.feature_column_names = [
            col for col in self.df_train if col not in ["cv", self.target_column_name]
        ]

        self.ua = UnivariateAnalysis(
            model_name=self.model_name,
            df_train=self.df_train,
            df_val=self.df_val,
            target_column_name=self.target_column_name,
            feature_column_names=self.feature_column_names,
            cv_folds=self.cv_folds,
            has_time_series_structure=self.has_time_series_structure,
        )
