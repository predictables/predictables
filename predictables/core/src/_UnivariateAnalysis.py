import datetime
import os
from typing import List, Optional

import numpy as np
import pandas as pd
import polars as pl

from predictables.univariate import Univariate
from predictables.util import DebugLogger, Report, to_pd_df, tqdm
from predictables.util.report.src._segment_features_for_report import (
    Segment,
    segment_features_for_report,
)

dbg = DebugLogger(working_file="_UnivariateAnalysis.py")
current_date = datetime.datetime.now()


class UnivariateAnalysis:
    cv_folds: pd.Series

    def __init__(
        self,
        model_name: str,
        df_train: pd.DataFrame,
        df_val: pd.DataFrame,
        target_column_name: str,
        feature_column_names: List[str],
        has_time_series_structure: bool,
        cv_column_name: Optional[str] = None,
        cv_folds: Optional[pd.Series] = None,
    ):
        dbg.msg("Initializing UnivariateAnalysis class - UA0001")  # debug only
        self.model_name = model_name
        self.df = df_train
        self.df_val = df_val
        self.target_column_name = target_column_name
        self.feature_column_names = feature_column_names
        self.cv_column_name = cv_column_name if cv_column_name is not None else "cv"
        self.cv_folds = (
            to_pd_df(self.df)[cv_column_name] if cv_folds is None else cv_folds
        )
        self.has_time_series_structure = has_time_series_structure

        feature_list = []
        for col in tqdm(
            self.feature_column_names,
            f"Performing univariate analysis on {len(self.feature_column_names)} "
            "features",
        ):
            obj_name = (
                col.lower()
                .replace(" ", "_")
                .replace("-", "_")
                .replace("/", "_")
                .replace("(", "")
                .replace(")", "")
            )
            setattr(
                self,
                obj_name,
                Univariate(self.df, self.df_val, "cv", col, self.target_column_name),
            )
            feature_list.append(obj_name)
            try:
                dbg.msg(
                    f"results for feature {col}: "
                    "{getattr(self, obj_name).results.head()} | UA0001a"
                )  # debug only
            except AttributeError:
                dbg.msg(
                    f"No results attribute found for feature {col} | UA0001b"
                )  # debug only
        self._feature_list = feature_list

    def get_features(self):
        dbg.msg("Getting features - UA0002")  # debug only
        return self._feature_list

    def _get_file_stem(
        self,
        filename: Optional[str] = None,
        default: str = "Univariate Analysis Report",
    ) -> str:
        """
        Helper function to get the file stem from a filename.

        Parameters
        ----------
        filename : str, optional
            The name of the file. The default is None.
        default : str, optional
            The default name of the file. The default is "Univariate Analysis Report".

        Returns
        -------
        str
            The file stem (eg the filename without the extension)

        Examples
        --------
        >>> _get_file_stem("Univariate Analysis Report.pdf")
        "Univariate Analysis Report"

        >>> _get_file_stem("Univariate Analysis Report")
        "Univariate Analysis Report"

        >>> _get_file_stem("Different Report.pdf", "Univariate Analysis Report")
        "Different Report"

        >>> _get_file_stem("Univariate Analysis Report.pdf", "Different Report")
        "Univariate Report"

        >>> _get_file_stem(None, "Univariate Analysis Report")
        "Univariate Analysis Report"
        """
        dbg.msg("Getting file stem - UA0003")
        if filename is not None:
            dbg.msg(
                f"Filename ({filename}) was passed to _get_file_stem - UA0004"
            )  # debug only
            file_stem, _ = os.path.splitext(filename)
            dbg.msg(
                f"File stem ({file_stem}) was extracted from filename ({filename}) "
                "and should not have an extension - UA0005"
            )  # debug only
            return file_stem

        return default

    def _rpt_filename(
        self,
        file_stem: Optional[str] = None,
        start_num: Optional[int] = None,
        end_num: Optional[int] = None,
        default: str = "Univariate Analysis Report",
    ) -> str:
        """Helper function to get the file name from a filename."""
        dbg.msg("Start of _rpt_filename - UA0006")  # debug only
        dbg.msg(
            f"Parameters: | file_stem: {file_stem} | start_num: {start_num} "
            f"| end_num: {end_num} | UA0006a"
        )  # debug only
        if file_stem is not None and (start_num is None or end_num is None):
            dbg.msg(
                f"File stem ({file_stem}) is not None and either start_num "
                f"({start_num}) or end_num ({end_num}) is None, so returning "
                f"'{file_stem}.pdf' | UA0006b"
            )  # debug only
            return file_stem + ".pdf"
        if start_num is not None and end_num is not None:
            dbg.msg(
                f"Both start_num ({start_num}) and end_num ({end_num}) are not None, "
                f"so returning '{file_stem}_{start_num+1}_{end_num+1}.pdf' | UA0006c"
            )
            return f"{file_stem}_{start_num+1}_{end_num+1}.pdf"
        else:
            dbg.msg(f"Returning default ({default}) | UA0006d")
            return default

    @staticmethod
    def _build_desc(total_features: int, max_per_file: int) -> str:
        return (
            f"Building {total_features} univariate analysis reports,"
            f"and packaging in increments of {max_per_file}"
        )

    def _segment_features(
        self, features: List[str], max_per_file: int
    ) -> List[Segment]:
        """
        Segments features into chunks for report generation.

        Parameters
        ----------
        features : list
            The list of features to segment.
        max_per_file : int
            The maximum number of features to include in a single report.

        Returns
        -------
        list
            The list of segments.

        Examples
        --------
        >>> _segment_features(["a", "b", "c", "d", "e"], 2)
        [
            Segment(start=0, end=2, n_features=2),
            Segment(start=2, end=4, n_features=2),
            Segment(start=4, n_features=1),
        ]

        >>> _segment_features(["a", "b", "c", "d", "e"], 4)
        [
            Segment(start=0, end=4, n_features=4),
            Segment(start=4, n_features=1),
        ]

        >>> _segment_features(["a", "b", "c", "d", "e"], 3)
        [
            Segment(start=0, end=3, n_features=3),
            Segment(start=3, n_features=2),
        ]

        """
        return segment_features_for_report(features, max_per_file)

    def _add_to_report(self, rpt: Report, feature: str) -> Report:
        ua = getattr(self, feature)
        return ua._add_to_report(rpt)

    def _generate_segment_report(
        self, segment: dict, filestem_: str, margins_: List[float]
    ):
        """Generates a report for a specific segment of features."""
        filename_ = self._rpt_filename(
            filestem_, segment["file_num_start"], segment["file_num_end"]
        )
        rpt = self._rpt_title_page(filename_, margins_)
        rpt = self._rpt_overview_page(
            rpt, segment["file_num_start"], segment["file_num_end"]
        )

        for feature in tqdm(
            segment["features"],
            desc=(
                f"Building report for features {segment['file_num_start']}"
                f" to {segment['file_num_end']}"
            ),
        ):
            rpt = self._add_to_report(rpt, feature=feature)

        rpt.build()

    def build_report(
        self,
        filename: Optional[str] = None,
        margins: Optional[List[float]] = None,
        max_per_file: int = 25,
    ) -> None:
        """
        Builds a report for the univariate analysis.

        Parameters
        ----------
        filename : str, optional
            The name of the file to save the report. The default is None.
        margins : list, optional
            The margins of the report. The default is None.
        max_per_file : int, optional
            The maximum number of features to include in a single report. The default
            is 25.

        Returns
        -------
        None
            Creates a report in the current working directory.
        """
        # Handle the case when None is passed as the filename
        filestem_ = self._get_file_stem(filename)
        features = self._sort_features_by_ua().index.tolist()
        # Handle the case when None is passed to the margins
        margins_: List[float] = margins if margins is not None else [0.5, 0.5, 0.5, 0.5]
        filename_: str

        if len(features) > max_per_file:
            files = segment_features_for_report(features, max_per_file)

            i = 0
            counter = 0
            start = files[i].start
            end = files[i].end
            fn = self._rpt_filename(filestem_, start, end)
            filename_ = f"{fn}"
            rpt = self._rpt_title_page(filename_, margins_)
            rpt = self._rpt_overview_page(rpt, files[i].start, files[i].end)
            for X in tqdm(features, self._build_desc(len(features), max_per_file)):
                rpt = getattr(self, X)._add_to_report(rpt)
                counter += 1

                if counter == max_per_file:
                    rpt.build()
                    i += 1
                    start = files[i].start
                    end = files[i].end
                    fn = self._rpt_filename(filestem_, start, end)
                    filename_ = f"{fn}"
                    rpt = self._rpt_title_page(filename_, margins_)
                    rpt = self._rpt_overview_page(rpt, files[i].start, files[i].end)
                    counter = 0

        else:
            # Handle the case when no filename is passed
            filename_ = f"{self._rpt_filename(filestem_)}"
            rpt = self._rpt_title_page(filename_, margins_)
            rpt = self._rpt_overview_page(rpt, 0, len(features) - 1)
            for X in tqdm(
                self.feature_column_names,
                f"Building {len(features)} univariate analysis reports",
            ):
                rpt = getattr(self, X)._add_to_report(rpt)

        rpt.build()

    def _rpt_overview_page(self, rpt: Report, first_idx: int, last_idx: int) -> Report:
        overview_df = self._sort_features_by_ua().iloc[first_idx : last_idx + 1]

        # Reformat to be a percentage with one decimal
        overview_df = overview_df.map(
            lambda x: (f"{np.round(x, 3):.1%}" if x < 1 else f"{np.round(x, 3):.1f}")
        )
        overview_df.index = overview_df.index.map(lambda x: x.replace("_", " ").title())

        return (
            rpt.h1("Overview")
            .h2(f"{self.model_name} Univariate Analysis Report")
            .p(
                "These sorted results for the features in this report indicate"
                " the average cross-validated test scores for each feature, "
                "if it were used as the only predictor in a simple linear "
                "model. "
                "Keep in mind that these results are based on the "
                "average, without considering the standard deviation. "
                "This means that the results are not necessarily the best "
                "predictors, but they are the best on average, and provide"
                "a fine starting point for grouping those predictors that "
                "are on average better than others. "
                "This means that nothing was done to account for possible "
                "sampling variability in the sorted results. "
                "This is a limitation of the univariate analysis, so it "
                "is important to keep this in mind when interpreting the "
                "results. "
                "It is also important to consider further that depending "
                "on the purpose of the model, the most appropriate features "
                "may not be the ones with the highest average test scores, "
                "if a different metric is more important."
            )
            .p(
                "In particular, this should not be taken as an opinion "
                "(actuarial or otherwise) regarding the most appropriate "
                "features to use in a model, but it rather provides a "
                "starting point for further analysis."
            )
            .spacer(0.125)
            .table(overview_df)
            .spacer(0.125)
            .caption(
                "This table shows an overview of the results for the "
                "variables in this file, representing those whose average test"
                f" score are ranked between {first_idx+1} and {last_idx+1} "
                f"of the variables passed to the {self.model_name}."
            )
            .page_break()
        )

    def _rpt_methodology_page(self, rpt: Report) -> Report:
        return (
            rpt.h1("Methodology")
            .h2(f"{self.model_name} Univariate Analysis Report")
            .h3("Introduction")
            .p("The univariate analysis is a simple method to evaluate")
            .page_break()
        )

    def _rpt_title_page(
        self,
        filename: Optional[str] = None,
        margins: Optional[List[float]] = None,
        date_of_report: datetime.datetime = current_date,
    ):
        rpt = Report(
            filename if filename is not None else "univariate_report.pdf",
            margins=margins if margins is not None else [0.5, 0.5, 0.5, 0.5],
        )
        date = rpt.date_(date_of_report)

        return (
            rpt.spacer(3)
            .h2(f"{self.model_name} Univariate Analysis Report")
            .style("h3", fontName="Helvetica")
            .h3(date)
            .page_break()
        )

    def _sort_features_by_ua(self):
        cols = []
        total_df = []
        for col in self.feature_column_names:
            if hasattr(self, col):
                ua = getattr(self, col)
            else:
                ua = Univariate(
                    self.df,
                    "cv",
                    col,
                    self.target_column_name,
                    time_series_validation=self.has_time_series_structure,
                )
            cols.append(col)
            total_df.append(
                ua.results.select(
                    [
                        pl.col("feature").alias("Feature"),
                        pl.col("acc_test").alias("Accuracy"),
                        pl.col("precision_test").alias("Precision"),
                        pl.col("recall_test").alias("Recall"),
                        pl.col("auc_test").alias("AUC"),
                        pl.col("f1_test").alias("F1"),
                        pl.col("mcc_test").alias("MCC"),
                        (
                            pl.col("acc_test")
                            + pl.col("precision_test")
                            + pl.col("recall_test")
                            + pl.col("auc_test")
                            + pl.col("f1_test")
                            + pl.col("mcc_test")
                        )
                        .truediv(6.0)
                        .alias("Ave."),
                    ]
                )
                .sort(
                    "Ave.",
                    descending=True,
                )
                .collect()
                .to_pandas()
                .set_index("Feature")
            )
        df = pd.concat(total_df).T
        return df.sort_values("Ave.", ascending=False)
