"""Implement the UnivariateFeatureEvaluator class."""

from dataclasses import dataclass
import polars as pl
from predictables.univariate import Univariate
from predictables.core.src.univariate_feature_evaluator.base_feature_evaluator import (
    BaseFeatureEvaluator,
)
import logging

@dataclass
class UnivariateFeatureEvaluator(BaseFeatureEvaluator):
    """Evaluate features for univariate analysis."""

    def evaluate_feature(self, feature: str) -> None:
        """Evaluate a feature using univariate analysis.

        Parameters
        ----------
        feature : str
            The feature to evaluate.
        """
        try:
            feature_df = self.df.filter(pl.col(feature).is_not_null())
            feature_val_df = self.df_val.filter(pl.col(feature).is_not_null())

            univariate = Univariate(
                feature_df,
                feature_val_df,
                self.cv_column_name,
                feature,
                self.target_column_name,
                self.time_series_validation,
            )
            self.add_result(univariate.results)
        except Exception as e:
            logging.error(f"Error evaluating feature {feature}: {e}")
            raise