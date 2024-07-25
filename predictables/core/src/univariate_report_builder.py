"""Builds a univariate analysis report for a given model and dataset."""

from __future__ import annotations
import polars as pl
from dataclasses import dataclass
from tqdm import tqdm
from typing import List

from predictables.util import Report
from predictables.univariate import Univariate
from datetime import datetime


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


@dataclass
class UnivariateReportBuilder:
    """Builds a univariate analysis report for a given model and dataset."""

    sorted_features: pl.LazyFrame
    model_name: str

    def build_report(
        self, filename: str | None, margins: List[float] | None, max_per_file: int
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
