import datetime
import os
from typing import List, Optional

import pandas as pd  # type: ignore
from tqdm import tqdm  # type: ignore

from predictables.univariate import Univariate
from predictables.util import Report

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
        cv_column_name: str,
        has_time_series_structure: bool,
    ):
        self.model_name = model_name
        self.df = df_train
        self.df_val = df_val
        self.target_column_name = target_column_name
        self.feature_column_names = feature_column_names
        self.cv_column_name = cv_column_name
        self.cv_folds = self.df[cv_column_name]
        self.has_time_series_structure = has_time_series_structure

        feature_list = []
        for col in tqdm(
            self.feature_column_names,
            f"Performing univariate analysis on {len(self.feature_column_names)} features",
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
        self._feature_list = feature_list

    def get_features(self):
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

        >>> _get_file_stem("Different Analysis Report.pdf", "Univariate Analysis Report")
        "Different Analysis Report"

        >>> _get_file_stem("Univariate Analysis Report.pdf", "Different Analysis Report")
        "Univariate Analysis Report"

        >>> _get_file_stem(None, "Univariate Analysis Report")
        "Univariate Analysis Report"
        """
        if filename is not None:
            file_stem, _ = os.path.splitext(filename)
            return file_stem

        return default

    def _rpt_filename(
        self,
        file_stem: Optional[str] = None,
        default: str = "Univariate Analysis Report",
        start_num: Optional[int] = None,
        end_num: Optional[int] = None,
    ) -> str:
        """Helper function to get the file name from a filename."""
        if file_stem is not None and (start_num is None or end_num is None):
            return file_stem + ".pdf"
        if start_num is not None and end_num is not None:
            return f"{file_stem}_{start_num}_{end_num}.pdf"
        else:
            return default

    @staticmethod
    def _build_desc(total_features: int, max_per_file: int) -> str:
        return (
            f"Building {total_features} univariate analysis reports,"
            f"and packaging in increments of {max_per_file}"
        )

    def _segment_features(self, features: List[str], max_per_file: int) -> List[dict]:
        """Segments features into chunks for report generation."""
        segments = []
        for i in range(0, len(features), max_per_file):
            start_index = i
            end_index = min(i + max_per_file, len(features))
            segment = {
                "file_num_start": start_index + 1,
                "file_num_end": end_index,
                "features": features[start_index:end_index],
            }
            segments.append(segment)
        return segments

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
            desc=f"Building report for features {segment['file_num_start']} to {segment['file_num_end']}",
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
            The maximum number of features to include in a single report. The default is 25.

        Returns
        -------
        None
            Creates a report in the current working directory.
        """
        # Handle the case when None is passed as the filename
        filestem_ = self._get_file_stem(filename)
        features = self._sort_features_by_ua().index.tolist()
        if len(features) > max_per_file:
            more_files = True
            files = []
            rem_features = features.copy()
            i = 0
        else:
            more_files = False

        while more_files:
            files.append(
                {
                    "file_num_start": i * max_per_file + 1,
                    "file_num_end": min(
                        (i + 1) * max_per_file,
                        (i * max_per_file) + len(rem_features) - 1,
                    ),
                    "features": rem_features[: min(max_per_file, len(rem_features))],
                }
            )

            rem_features = rem_features[min(max_per_file, len(rem_features)) :]
            i += 1
            if len(rem_features) == 0:
                more_files = False

        # Handle the case when None is passed to the margins
        margins_: List[float] = margins if margins is not None else [0.5, 0.5, 0.5, 0.5]
        filename_: str
        i = 0
        if len(features) > max_per_file:
            # Handle the case when no filename is passed
            counter = 0
            filename_ = f"{self._rpt_filename(filestem_,start_num=files[i]['file_num_start'],end_num=files[i]['file_num_end'])}"
            rpt = self._rpt_title_page(filename_, margins_)
            rpt = self._rpt_overview_page(
                rpt, files[i]["file_num_start"], files[i]["file_num_end"]
            )
            for X in tqdm(features, self._build_desc(len(features), max_per_file)):
                rpt = getattr(self, X)._add_to_report(rpt)
                counter += 1

                if counter == max_per_file:
                    rpt.build()
                    i += 1
                    filename_ = f"{self._rpt_filename(filestem_,start_num=files[i]['file_num_start'],end_num=files[i]['file_num_end'])}"
                    rpt = self._rpt_title_page(filename_, margins_)
                    rpt = self._rpt_overview_page(
                        rpt, files[i]["file_num_start"], files[i]["file_num_end"]
                    )
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
            lambda x: f"{x:.1%}" if x > 0.0001 else f"{x:.2e}"
        )

        return (
            rpt.h1("Overview")
            .h2(f"{self.model_name} Univariate Analysis Report")
            .h3(
                "These sorted results for the features in this report indicate the average cross-validated test scores for each feature, if it were used as the only predictor in a simple linear model. Keep in mind that these results are based on the average, without considering the standard deviation. This means that the results are not necessarily the best predictors, but they are the best on average, and provide a fine starting point for grouping those predictors that are on average better than others. This means that nothing was done to account for possible sampling variability in the sortied results. This is a limitation of the univariate analysis, and it is important to keep this in mind when interpreting the results. It is also important to consider further that depending on the purpose of the model, the most appropriate features may not be the ones with the highest average test scores, if a different metric is more important."
            )
            .h3(
                "In particular, this should not be taken as an opinion (actuarial or otherwise) regarding the most appropriate features to use in a model, but it rather provides a starting point for further analysis."
            )
            .spacer(0.125)
            .table(overview_df)
            .spacer(0.125)
            .caption(
                f"This table shows an overview of the results for the variables in this file, representing those whose average test score are ranked between {first_idx+1} and {last_idx+1} of the variables passed to the {self.model_name}."
            )
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
        results = []
        total_df = []
        for col in tqdm(
            self.feature_column_names,
            "Sorting features by their relevance based on an univariate analysis",
        ):
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
            results.append(ua.pareto_sort_vector)
            total_df.append(
                [
                    ua.acc_test,
                    ua.precision_test,
                    ua.recall_test,
                    ua.auc_test,
                    ua.f1_test,
                    ua.mcc_test,
                ]
            )
        df = pd.DataFrame(
            total_df,
            columns=["Accuracy", "Precision", "Recall", "AUC", "F1", "MCC"],
            index=cols,
        )
        df["Ave."] = df.mean(axis=1).values
        return df.sort_values("Ave.", ascending=False)
