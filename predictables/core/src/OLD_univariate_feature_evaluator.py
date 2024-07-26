"""Univariate feature evaluator."""

from __future__ import annotations

import polars as pl

from predictables.univariate import Univariate
from dataclasses import dataclass, field
from typing import List
from predictables.core.src.univariate_config import UnivariateConfig


@dataclass
class UnivariateFeatureEvaluator:
    """Evaluate features for univariate analysis."""

    config = UnivariateConfig

    @property
    def df(self) -> pl.LazyFrame:
        """Return the training data."""
        return self.config.df_train

    @property
    def df_val(self) -> pl.LazyFrame:
        """Return the validation data."""
        return self.config.df_val

    @property
    def cv_column_name(self) -> str:
        """Return the cross-validation column name."""
        return self.config.cv_column_name

    @property
    def target_column_name(self) -> str:
        """Return the target column name."""
        return self.config.target_column_name

    @property
    def time_series_validation(self) -> bool:
        """Return whether to use time series validation."""
        return self.config.time_series_validation

    results: List[Univariate] = field(default_factory=list)

    def evaluate_feature(self, feature: str) -> None:
        """Evaluate a feature."""
        univariate = Univariate(
            self.df.filter(pl.col(feature).is_not_null()),
            self.df_val.filter(pl.col(feature).is_not_null()),
            self.cv_column_name,
            feature,
            self.target_column_name,
            self.time_series_validation,
        )
        self.results.append(univariate.results)

    def sort_features(self) -> pl.LazyFrame:
        """Sort features by AUC."""
        return pl.concat(self.results).sort("AUC", descending=True)
