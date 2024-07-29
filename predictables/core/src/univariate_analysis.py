"""Perform a full univariate analysis."""

from __future__ import annotations
from tqdm import tqdm
from predictables.core.src.univariate_feature_evaluator import (
    UnivariateFeatureEvaluator,
    FeatureEvaluatorInterface,
)
from predictables.core.src.univariate_config import UnivariateConfigInterface
from predictables.core.src.univariate_feature_transformer.base_feature_transformer import (
    FeatureTransformerInterface,
    UnivariateFeatureTransformer,
)
from predictables.core.src.univariate_report_builder import (
    UnivariateReportBuilder,
    ReportBuilderInterface,
)
from predictables.util import Report, to_pl_lf


class UnivariateAnalysis:
    """Perform a full univariate analysis."""

    def __init__(
        self,
        config: UnivariateConfigInterface,
        feature_transformer: FeatureTransformerInterface | None = None,
        feature_evaluator: FeatureEvaluatorInterface | None = None,
        report_builder: ReportBuilderInterface | None = None,
    ) -> None:
        self._config = config
        self._feature_transformer = feature_transformer
        self._feature_evaluator = feature_evaluator
        self._report_builder = report_builder

    @property
    def config(self) -> UnivariateConfigInterface:
        """Return the univariate config."""
        return self._config

    @property
    def feature_transformer(self) -> FeatureTransformerInterface:
        """Return the feature transformer."""
        if self._feature_transformer is None:
            self._feature_transformer = UnivariateFeatureTransformer(self.config)

        return (
            UnivariateFeatureTransformer(self.config)
            if self._feature_transformer is None
            else self._feature_transformer
        )

    @property
    def feature_evaluator(self) -> FeatureEvaluatorInterface:
        """Return the feature evaluator."""
        if self._feature_evaluator is None:
            self._feature_evaluator = UnivariateFeatureEvaluator(self.config)

        return (
            UnivariateFeatureEvaluator(self.config)
            if self._feature_evaluator is None
            else self._feature_evaluator
        )

    @property
    def report_builder(self) -> ReportBuilderInterface:
        """Return the report builder."""
        if self._report_builder is None:
            self._report_builder = UnivariateReportBuilder(self.config)

        return (
            UnivariateReportBuilder(self.config)
            if self._report_builder is None
            else self._report_builder
        )

    def perform_analysis(self) -> list[str]:
        """Perform the univariate analysis for each feature."""
        for feature in tqdm(
            self.config.features, desc="Performing univariate analysis"
        ):
            self.feature_evaluator.evaluate_feature(feature)
        return self.feature_evaluator.sort_features()

    def build_report(
        self,
        filename: str | None = None,
        margins: tuple[int, int] | None = None,
        max_per_file: int = 25,
    ) -> None:
        """Build the univariate analysis report."""
        self.report_builder.build_report(filename, margins, max_per_file)
