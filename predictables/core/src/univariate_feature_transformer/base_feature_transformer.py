"""Transform features for univariate analysis."""

from __future__ import annotations

import polars as pl
from typing import List
from dataclasses import dataclass

from predictables.core.src.univariate_config import UnivariateConfig


@dataclass
class BaseFeatureTransformer:
    """Base class for feature transformers."""

    config: UnivariateConfig

    @property
    def df(self) -> pl.LazyFrame:
        """Return the training data."""
        return self.config.df_train

    @df.setter
    def df(self, df: pl.LazyFrame) -> None:
        """Set the training data."""
        self.config.df_train = df

    @property
    def df_val(self) -> pl.LazyFrame:
        """Return the validation data."""
        return self.config.df_val

    @df_val.setter
    def df_val(self, df: pl.LazyFrame) -> None:
        """Set the validation data."""
        self.config.df_val = df

    @property
    def feature_column_names(self) -> List[str]:
        """Return the feature column names."""
        return self.config.feature_column_names

    def transform_features(self) -> List[str]:
        """Transform features and return the list of transformed feature names."""
        raise NotImplementedError("Subclasses should implement this method.")
