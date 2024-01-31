from typing import Optional, Tuple

import pandas as pd
from tqdm import tqdm

from predictables.univariate import Univariate
from predictables.util import Report, to_pd_df


class UnivariateAnalysis:
    def __init__(
        self,
        df_train,
        target_column_name,
        feature_column_names,
        cv_folds,
        has_time_series_structure,
    ):
        self.df = to_pd_df(df_train).assign(cv=cv_folds)
        self.target_column_name = target_column_name
        self.feature_column_names = feature_column_names
        self.has_time_series_structure = has_time_series_structure

        feature_list = []
        for col in tqdm(
            self.feature_column_names,
            f"Performing univariate analysis on {len(self.feature_column_names)} features.",
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
                Univariate(self.df, "cv", col, self.target_column_name),
            )
            feature_list.append(obj_name)
        self._feature_list = feature_list

    def get_features(self):
        return self._feature_list

    def build_report(
        self,
        filename: Optional[str] = None,
        margins: Optional[Tuple[float, float, float, float]] = None,
        max_per_file: int = 25,
    ):
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
                    "file_num_end": (i + 1) * max_per_file,
                    "features": rem_features[: min(max_per_file, len(rem_features))],
                }
            )

            rem_features = rem_features[min(max_per_file, len(rem_features)) :]
            i += 1
            if len(rem_features) == 0:
                more_files = False

        if len(features) > max_per_file:
            counter = 0
            i = 0
            filename = filename.split(".")
            filename = filename[0]
            rpt = self._rpt_title_page(
                filename=f"{filename}_{files[i]['file_num_start']}_{files[i]['file_num_end']}.pdf",
                margins=margins,
            )
            for X in tqdm(
                features,
                f"Building {len(features)} univariate analysis reports, and packaging in increments of {max_per_file}",
            ):
                rpt = getattr(self, X)._add_to_report(rpt)
                counter += 1

                if counter == max_per_file:
                    rpt.build()
                    i += 1
                    rpt = self._rpt_title_page(
                        filename=f"{filename}_{files[i]['file_num_start']}_{files[i]['file_num_end']}.pdf",
                        margins=margins,
                    )
                    counter = 0

        else:
            rpt = self._rpt_title_page(
                filename=f"{filename}.pdf",
                margins=margins,
            )
            for X in tqdm(
                features,
                f"Building {len(features)} univariate analysis reports.",
            ):
                rpt = getattr(self, X)._add_to_report(rpt)

        rpt.build()

    def _rpt_title_page(
        self,
        rpt: Optional[Report] = None,
        filename: Optional[str] = None,
        margins: Optional[Tuple[float, float, float, float]] = None,
    ):
        if margins is None:
            margins = [0.5, 0.5, 0.5, 0.5]
        if filename is None:
            filename = "univariate_report.pdf"
        if rpt is None:
            rpt = Report(filename, margins=margins)

        return (
            rpt.spacer(4)
            .h1("Univariate Analysis")
            .h2("Hit Ratio Model")
            .style("h3", fontName="Helvetica")
            .style("h4", fontName="Helvetica")
            .page_break()
        )

    def _sort_features_by_ua(self):
        cols = []
        results = []
        total_df = []
        for col in tqdm(self.feature_column_names):
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
