"""Base feature evaluator for univariate analysis."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List
import polars as pl
from predictables.core.src.univariate_config import UnivariateConfigInterface

import logging


@dataclass
class BaseFeatureEvaluator:
    """Base class for feature evaluators."""

    config: UnivariateConfigInterface
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
    def target(self) -> str:
        """Return the target column name."""
        return self.config.target_column_name

    @property
    def feature_column_names(self) -> List[str]:
        """Return the feature column names."""
        return self.config.feature_column_names

    @property
    def features(self) -> List[str]:
        """Return the feature column names."""
        return self.config.feature_column_names

    @property
    def time_series_validation(self) -> bool:
        """Return whether to use time series validation."""
        return self.config.time_series_validation

    results: List[pl.LazyFrame] = field(default_factory=list)

    def evaluate_feature(self, feature: str) -> None:
        """Evaluate a feature. To be implemented by subclasses."""
        raise NotImplementedError("Subclasses should implement this method.")

    def add_result(self, result: pl.LazyFrame) -> None:
        """Add evaluation result to the results list.

        Parameters
        ----------
        result : pl.LazyFrame
            The evaluation result to add.
        """
        self.results.append(result)

    def sort_features_by_metric(
        self, metric: str = "AUC", descending: bool = True
    ) -> pl.LazyFrame:
        """Sort features by a specified metric.

        Parameters
        ----------
        metric : str
            The metric to sort by (default is "AUC").
        descending : bool
            Sort order (default is True for descending).

        Returns
        -------
        pl.LazyFrame
            Sorted features by the specified metric.
        """
        if not self.results:
            logging.warning("No results to sort.")
            return pl.LazyFrame()
        try:
            return pl.concat(self.results).sort(metric, descending=descending)
        except Exception as e:
            logging.error(f"Error sorting features by {metric}: {e}")
            raise
