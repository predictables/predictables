"""Transform features for univariate analysis."""

from __future__ import annotations

import polars as pl
from typing import List
from dataclasses import dataclass

from univariate_config import UnivariateConfig


@dataclass
class UnivariateFeatureTransformer:
    """Transform features for univariate analysis."""

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

    def _feature_skewness(self, feature: str) -> float:
        """Return the skewness of a feature."""
        return self.df.select(pl.col(feature).skew()).collect().item()

    def transform_features(self) -> List[str]:
        """Transform features for univariate analysis."""
        transformed_features = []
        for feature in self.feature_column_names:
            transformed_features.append(feature)
            if self._feature_skewness(feature) > 0.5:
                self._apply_log_transform(feature)
                transformed_features.append(f"log1p_{feature}")
        return transformed_features

    def _apply_log_transform(self, feature: str) -> None:
        log_feature = f"log1p_{feature}"

        df = self.df.with_columns(pl.col(feature).log1p().alias(log_feature))
        self.df(df)

        df_val = self.df_val.with_columns(pl.col(feature).log1p().alias(log_feature))
        self.df_val(df_val)
