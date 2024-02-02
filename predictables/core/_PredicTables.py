from typing import List, Union

import pandas as pd
import polars as pl

from predictables.core.src._UnivariateAnalysis import UnivariateAnalysis
from predictables.util import to_pd_df, to_pd_s


class PredicTables:
    feature_column_names: List[str]

    def __init__(
        self,
        model_name: str,
        df_train: Union[pd.DataFrame, pl.DataFrame, pl.LazyFrame],
        df_val: Union[pd.DataFrame, pl.DataFrame, pl.LazyFrame],
        df_test: Union[pd.DataFrame, pl.DataFrame, pl.LazyFrame],
        target_column_name: str,
        cv_folds: Union[pl.Series, pd.Series],
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
            col for col in self.df_train if col not in ["cv", self.target_column_name]  # type: ignore
        ]

        self.ua = UnivariateAnalysis(
            self.model_name,
            self.df_train,
            self.df_val,
            self.target_column_name,
            self.feature_column_names,
            self.cv_folds,
            has_time_series_structure=self.has_time_series_structure,
        )
