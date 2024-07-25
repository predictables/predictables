"""Configuration for univariate analysis."""

from __future__ import annotations
from dataclasses import dataclass
from typing import List
import polars as pl


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
