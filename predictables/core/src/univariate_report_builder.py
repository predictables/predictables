"""Builds a univariate analysis report for a given model and dataset."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import List

import polars as pl
from tqdm import tqdm

from predictables.univariate import Univariate
from predictables.util import Report
from predictables.util.formatter import FormatterInterface, PercentFormatter

from predictables.core.src.univariate_config import UnivariateConfig, UnivariateConfigInterface
from predictables.core.src.univariate_feature_transformer import FeatureTransformerInterface


def _format_values(col: str) -> pl.Expr:
    """Format values for the report."""
    return (
        pl.when(pl.col(col) < 1)
        .then(
            (pl.col(col) * 100).cast(pl.Float64).round(decimals=1).cast(pl.Utf8) + "%"
        )
        .otherwise(pl.col(col).round(decimals=1).cast(pl.Utf8))
    )


def segment_features_for_report(
    features: list[str], max_per_file: int
) -> list[list[str]]:
    """Segment features into groups of max_per_file."""
    return [
        features[i : i + max_per_file] for i in range(0, len(features), max_per_file)
    ]

class UnivariateReportConfig:
    """Configuration options for a univariate report."""

    def __init__(
        self,
        filename: str | None,
        margins: List[float] | None,
        max_per_file: int = 25
    ):
        self._filename = filename
        self._margins = margins
        self._max_per_file = max_per_file

    @property
    def filename(self) -> str:
        """Return either the user-provided filename or a default if None is provided."""
        return (
            "univariate_analysis"
            if self._filename is None
            else self._filename
        )

    @filename.setter
    def filename(self, name: str) -> None:
        """Set the filename."""
        self._filename = name.replace(".pdf", "")

    @property
    def margins(self) -> str:
        """Return a tuple with the margins or a default if None is provided."""
        return (
            (0.5, 0.5, 0.5, 0.5)
            if self._margins is None
            else self._margins
        )

class UnivariateReportBuilder:
    """Builds a univariate analysis report for a given model and dataset."""

    def __init__(
        self,
        config: UnivariateConfigInterface,
        feature_evaluator: FeatureEvaluatorInterface
    ): 
        self._config = config
        self._feature_evaluator = feature_evaluator

    @property
    def config(self) -> UnivariateConfigInterface:
        """Return the configuration for the univariate analysis."""
        return self._config

    @property
    def feature_evaluator(self) -> FeatureEvaluatorInterface:
        """Return the feature evaluator."""


    # sorted_features: pl.LazyFrame
    # model_name: str

    def build_report(
        self
    ) -> None:
        """Build a univariate analysis report."""
        filestem = self._get_file_stem(filename)
        features = (
            self.sorted_features.select("Feature").collect().to_series().to_List()
        )
        margins = margins if margins else [0.5, 0.5, 0.5, 0.5]
        segments = self._segment_features(features, max_per_file)

        for segment in tqdm(segments, desc="Building reports"):
            self._generate_segment_report(segment, filestem, margins)

    def _get_file_stem(self, filename: str) -> str:
        """Return the file stem, eg 'Univariate Analysis Report'."""
        if filename:
            return filename.rsplit(".", 1)[0]
        return "Univariate Analysis Report"

    def _segment_features(
        self, features: List[str], max_per_file: int
    ) -> List[List[str]]:
        """Segment features into groups of max_per_file."""
        return [
            features[i : i + max_per_file]
            for i in range(0, len(features), max_per_file)
        ]

    def _generate_segment_report(
        self, segment: List[str], filestem: str, margins: List[float]
    ) -> None:
        """Generate a report for a segment of features."""
        filename = f"{filestem}_{segment[0]}_{segment[-1]}.pdf"
        rpt = Report(filename, margins)
        rpt = self._rpt_title_page(rpt)
        rpt = self._rpt_overview_page(rpt, segment)
        for feature in segment:
            rpt = self._add_to_report(rpt, feature)
        rpt.build()

    def _rpt_title_page(self, rpt: Report) -> Report:
        """Add the title page to the report."""
        date = rpt.date_(datetime.datetime.now())
        return (
            rpt.spacer(3)
            .h2(f"{self.model_name} Univariate Analysis Report")
            .style("h3", fontName="Helvetica")
            .h3(date)
            .page_break()
        )

    def _rpt_overview_page(self, rpt: Report, segment: List[str]) -> Report:
        """Add the overview page to the report."""
        overview_df = self.sorted_features.filter(pl.col("Feature").is_in(segment))
        formatted_df = overview_df.select(
            [pl.col("Feature").alias("Feature")]
            + [_format_values(col) for col in overview_df.columns if col != "Feature"]
        )
        return (
            rpt.h1("Overview")
            .h2(f"{self.model_name} Univariate Analysis Report")
            .table(formatted_df.collect().to_pandas().set_index("Feature"))
            .page_break()
        )

    def _add_to_report(self, rpt: Report, feature: str) -> Report:
        """Add a feature to the report."""
        ua = Univariate(
            self.df.filter(pl.col(feature).is_not_null()),
            self.df_val.filter(pl.col(feature).is_not_null()),
            self.cv_column_name,
            feature,
            self.target_column_name,
            self.time_series_validation,
        )
        return ua._add_to_report(rpt)  # noqa: SLF001
