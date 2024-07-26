"""Perform a full univariate analysis."""

from __future__ import annotations
from typing import List, Tuple

from predictables.util import to_pl_lf
from predictables.core.src.univariate_config import UnivariateConfig
from predictables.core.src.feature_transformers.base_feature_transformer import (
    UnivariateFeatureTransformer,
)
from predictables.core.src.univariate_feature_evaluator import (
    UnivariateFeatureEvaluator,
)
from predictables.core.src.univariate_report_builder import UnivariateReportBuilder
from dataclasses import dataclass
from tqdm import tqdm
import polars as pl

from predictables.util import Report


@dataclass
class UnivariateAnalysis:
    """Perform a full univariate analysis."""

    config: UnivariateConfig

    feature_transformer: UnivariateFeatureTransformer
    feature_evaluator: UnivariateFeatureEvaluator
    report_builder: UnivariateReportBuilder

    reports: List[Report]

    @property
    def df(self) -> pl.LazyFrame:
        """Return the training data."""
        return to_pl_lf(self.config.df_train)

    @df.setter
    def df(self, df: pl.LazyFrame) -> None:
        """Update the training data."""
        self.config.df_train = df

    @property
    def df_val(self) -> pl.LazyFrame:
        """Return the validation data."""
        return to_pl_lf(self.config.df_val)

    @df_val.setter
    def df_val(self, df: pl.LazyFrame) -> None:
        """Update the validation data."""
        self.config.df_val = df

    @property
    def cv_folds(self) -> pl.Series:
        """Return the cross-validation fold labels."""
        return (
            self.df.select([pl.col(self.config.cv_column_name)]).collect().to_series()
            if self.config.cv_folds is None
            else self.config.cv_folds
        )

    @cv_folds.setter
    def cv_folds(self, cv_folds: pl.Series) -> None:
        """Update the cross-validation fold labels."""
        self.config.cv_folds = cv_folds

    def __post_init__(self) -> None:
        """Initialize the univariate analysis."""
        self.feature_transformer = UnivariateFeatureTransformer(self.config)
        self.feature_evaluator = UnivariateFeatureEvaluator(self.config)

        self.features = self.feature_transformer.transform_features()
        self.reports = []

    def _initialize_cv_folds(self, cv_folds: pl.Series | None) -> pl.Series:
        """Return the cross-validation fold labels."""
        return (
            self.df.select([pl.col(self.config.cv_column_name)]).collect().to_series()
            if cv_folds is None
            else cv_folds
        )

    def perform_analysis(self) -> None:
        """Perform the univariate analysis for each feature."""
        for feature in tqdm(self.features, desc="Performing univariate analysis"):
            self.feature_evaluator.evaluate_feature(feature)
        self.sorted_features = self.feature_evaluator.sort_features()

    def build_report(
        self,
        filename: str | None = None,
        margins: Tuple[int, int] | None = None,
        max_per_file: int = 25,
    ) -> None:
        """Build the univariate analysis report."""
        self.report_builder.build_report(filename, margins, max_per_file)
