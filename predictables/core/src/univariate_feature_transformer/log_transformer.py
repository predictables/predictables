"""Feature transformer for log transformation."""

from __future__ import annotations

from predictables.core.src.univariate_feature_transformer.base_feature_transformer import BaseFeatureTransformer
import polars as pl
from typing import List
import logging

__all__ = ["LogTransformer"]

class LogTransformer(BaseFeatureTransformer):
    """Apply log transformation to features based on skewness."""

    def __init__(
        self, 
        skewness_threshold: float = 0.5
    ):
        self._skewness_threshold = skewness_threshold

    @property
    def skewness_threshold(self) -> float:
        """Return the skewness threshold for log transformation."""
        return self._skewness_threshold

    @skewness_threshold.setter
    def skewness_threshold(self, skewness_threshold: float) -> None:
        """Set the skewness threshold for log transformation."""
        self._skewness_threshold = skewness_threshold

    def _calculate_skewness(self, feature: str) -> float:
        """Calculate the skewness of a feature.

        Parameters
        ----------
        feature : str
            The name of the feature to calculate skewness for.

        Returns
        -------
        float
            The skewness value of the feature.
        """
        try:
            return self.df.select(pl.col(feature).skew()).collect().item()
        except Exception as e:
            logging.error(f"Error calculating skewness for feature {feature}: {e}")
            raise

    def _should_apply_log_transform(self, skewness: float) -> bool:
        """Determine if log transformation should be applied based on skewness.

        Parameters
        ----------
        skewness : float
            The skewness value of the feature.

        Returns
        -------
        bool
            True if log transformation should be applied, False otherwise.
        """
        return skewness > self.skewness_threshold

    def _apply_log_transform(self, feature: str) -> None:
        """Apply log1p transformation to a feature.

        Parameters
        ----------
        feature : str
            The name of the feature to transform.

        Returns
        -------
        None
            The `df` and `df_val` attributes are updated in place.
        """
        try:
            log_feature = f"log1p_{feature}"
            self.config.df(
                self.df.with_columns(
                    pl.col(feature).log1p().alias(log_feature)
                )
            )
            self.config.df_val(
                self.df_val.with_columns(
                    pl.col(feature).log1p().alias(log_feature)
                )
            )
        except Exception as e:
            logging.error(
                f"Error applying log transform to feature {feature}: {e}"
            )
            raise

    def transform_features(self) -> List[str]:
        """Transform features for univariate analysis.

        Returns
        -------
        List[str]
            A list of transformed feature names.
        """
        transformed_features = []
        for feature in self.config.features:
            transformed_features.append(feature)
            try:
                skewness = self._calculate_skewness(feature)
                if self._should_apply_log_transform(skewness):
                    self._apply_log_transform(feature)
                    transformed_features.append(f"log1p_{feature}")
            except Exception as e:
                logging.error(f"Error transforming feature {feature}: {e}")
                continue
        return transformed_features
