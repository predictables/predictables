"""Configuration for univariate analysis."""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Generator, Tuple
import polars as pl
import pandas as pd


@dataclass
class UnivariateConfig:
    """Configure options for a univariate analysis."""

    model_name: str
    df_train: pl.LazyFrame
    df_val: pl.LazyFrame
    target_column_name: str
    feature_column_names: List[str]
    time_series_validation: bool
    cv_column_name: str = "cv"
    cv_folds: pl.Series | None = None

    @property
    def df(self) -> pl.LazyFrame:
        """Return the training data."""
        return self.df_train

    @df.setter
    def df(self, df: pl.LazyFrame) -> None:
        """Set the training data."""
        self.df_train = df

    @property
    def df_val(self) -> pl.LazyFrame:
        """Return the validation data."""
        return self.df_val

    @df_val.setter
    def df_val(self, df: pl.LazyFrame) -> None:
        """Set the validation data."""
        self.df_val = df

    @property
    def features(self) -> List[str]:
        """Return the feature column names."""
        return self.feature_column_names

    @property
    def X_y_by_cv(self) -> Generator[tuple[pd.DataFrame, pd.DataFrame], None, None]:
        """Return the training and validation data as a generator."""
        for i in range(self.cv_folds.min(), self.cv_folds.max() + 1):
            df_train, df_val = (
                self.filter_for_time_series(self.df, i)
                if self.time_series_validation
                else self.filter_for_non_time_series(self.df, i)
            )
            X_train = (
                df_train.drop([self.target_column_name, self.cv_column_name])
                .collect()
                .to_pandas()
            )
            y_train = (
                df_train.select(self.target_column_name).collect().to_numpy().ravel()
            )
            X_test = (
                df_val.drop([self.target_column_name, self.cv_column_name])
                .collect()
                .to_pandas()
            )
            y_test = df_val.select(self.target_column_name).collect().to_numpy().ravel()
            yield X_train, X_test, y_train, y_test

    def filter_for_time_series(
        self, X: pl.LazyFrame, fold: int
    ) -> Tuple[pl.LazyFrame, pl.LazyFrame]:
        """Filter the training and testing data for a particular fold, given time series validation."""
        return X.filter(pl.col(self.cv_column_name) < fold), X.filter(
            pl.col(self.cv_column_name) == fold
        )

    def filter_for_non_time_series(
        self, X: pl.LazyFrame, fold: int
    ) -> Tuple[pl.LazyFrame, pl.LazyFrame]:
        """Filter the training and testing data for a particular fold, given non-time series validation."""
        return X.filter(pl.col(self.cv_column_name) != fold), X.filter(
            pl.col(self.cv_column_name) == fold
        )
