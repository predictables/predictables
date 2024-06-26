from __future__ import annotations

import datetime

import pandas as pd
import polars as pl

from predictables.univariate import Univariate
from predictables.util import Report, fmt_col_name, get_column_dtype, to_pl_lf, tqdm
from predictables.util.report.src._segment_features_for_report import (
    Segment,
    segment_features_for_report,
)

current_date = datetime.datetime.now()


def _format_values(col: str) -> pl.Expr:
    return (
        pl.when(pl.col(col) < 1)
        .then(
            (pl.col(col) * 100).cast(pl.Float64).round(decimals=1).cast(pl.Utf8) + "%"
        )
        .otherwise(pl.col(col).round(decimals=1).cast(pl.Utf8))
    )


class UnivariateAnalysis:
    """
    Perform univariate analysis on a dataset to evaluate the impact of individual features on a target variable.

    This class facilitates the exploration and evaluation of each feature's predictive power and importance
    by creating Univariate objects for each feature in the dataset. It supports both cross-validation and
    time-series validation strategies to assess feature performance. The class also provides functionalities
    to rank features based on their performance metrics and to format feature names to be used as valid
    Python attribute names.

    After initializing the class, the user can access the Univariate objects for each feature to view the
    results of the univariate analysis. The user can also generate a report containing the results of the
    univariate analysis for all features in the dataset.

    Attributes
    ----------
    model_name : str
        Name of the model for which the univariate analysis is being performed.
    df : pd.DataFrame
        Training dataset containing features and the target variable.
    df_val : pd.DataFrame
        Validation dataset used for evaluating feature performance.
    target_column_name : str
        Name of the target variable in the dataset.
    feature_column_names : list[str]
        list of names of the features to be analyzed.
    cv_column_name : str | None
        Name of the column used for cross-validation splitting. Defaults to "cv" if not provided.
    cv_folds : Optional[pd.Series]
        Series containing cross-validation fold identifiers for each row in the dataset. If not provided,
        it is assumed that the cv_column_name in the dataset is used.
    time_series_validation : bool
        Indicates whether to use time series validation strategy for feature evaluation.
    right_skewness_threshold : float
        The threshold for right skewness. Defaults to 0.5. If the skewness of a feature is greater than this
        threshold, several transformed variables will be added to the analysis:
            - box-cox transformation
                - handles strictly positive features
                - log-transform is a special case of the box-cox transform
            - yeo-johnson transformation
                - handles strictly positive and strictly negative features
                - a generalization of the box-cox transform
            - quantile transformation (standard normal)
                - handles all features
    left_skewness_threshold : float
        The threshold for left skewness. Defaults to -0.5. If the skewness of a feature is less than this
        threshold, several transformed variables will be added to the analysis:
            - box-cox transformation, reflected over the y-axis
                - handles strictly negative features
            - yeo-johnson transformation, reflected over the y-axis
                - handles strictly positive and strictly negative features
            - quantile transformation (standard normal), reflected over the y-axis
                - handles all features

    Methods
    -------
    __init__(self, model_name, df_train, df_val, target_column_name, feature_column_names, time_series_validation, cv_column_name=None, cv_folds=None):
        Initializes the UnivariateAnalysis class with the dataset, features, and validation settings.
    _sort_features_by_ua():
        Sorts features based on their average performance metrics.
    get_features():
        Returns a list of feature names that have been analyzed.
    _get_file_stem(filename=None, default="Univariate Analysis Report"):
        Helper function to get the file stem from a filename.
    _rpt_filename(file_stem=None, start_num=None, end_num=None, default="Univariate Analysis Report"):
        Helper function to get the file name from a filename.
    _build_desc(total_features, max_per_file):
        Helper function to build a description for the report generation process to keep the user informed
        about the progress of the report generation.
    _segment_features(features, max_per_file):
        Segments features into chunks for report generation.
    _add_to_report(rpt, feature):
        Adds the results of the univariate analysis for a feature to the report.
    _generate_segment_report(segment, filestem_, margins_):
        Generates a report for a specific segment of features.
    build_report(filename=None, margins=None, max_per_file=25):
        Builds a report for the univariate analysis.

    Examples
    --------
    >>> df_train = pd.DataFrame(
    ...     {"A": [1, 2, 3], "B": [4, 5, 6], "target": [0, 1, 1]}
    ... )
    >>> df_val = pd.DataFrame(
    ...     {"A": [2, 3, 4], "B": [5, 6, 7], "target": [1, 0, 1]}
    ... )
    >>> ua = UnivariateAnalysis(
    ...     "MyModel", df_train, df_val, "target", ["A", "B"], False
    ... )
    >>> ua.A.results.head()
    [1] feature  |  acc_test  |  precision_test  |  recall_test  |  auc_test  |  f1_test  |  mcc_test
    [2] A
    [3] B
    >>> ua.get_features()
    ['A', 'B']
    >>> ua.build_report("Univariate Analysis Report.pdf")
    >>> # This will generate a report containing the results of the univariate analysis for the features in the dataset.
    """

    cv_folds: pd.Series

    def __init__(
        self,
        model_name: str,
        df_train: pl.LazyFrame,
        df_val: pl.LazyFrame,
        target_column_name: str,
        feature_column_names: list[str],
        time_series_validation: bool,
        cv_column_name: str | None = None,
        cv_folds: pl.Series | None = None,
    ):
        """Initialize the UnivariateAnalysis class.

        Initializes the UnivariateAnalysis class with the dataset, features,
        and validation settings.

        Parameters
        ----------
        model_name : str
            Name of the model for which the univariate analysis is being performed.
        df_train : pl.LazyFrame
            The training dataset containing features and the target variable.
        df_val : pl.LazyFrame
            The validation dataset used for evaluating feature performance.
        target_column_name : str
            The name of the target variable in the dataset.
        feature_column_names : list[str]
            A list of names of the features to be analyzed.
        time_series_validation : bool
            Indicates whether to use a time series validation strategy for feature evaluation.
        cv_column_name : str | None, optional
            The name of the column used for cross-validation splitting, by default None. If None,
            a default column name "cv" is assumed.
        cv_folds : pl.Series | None, optional
            A series containing cross-validation fold identifiers for each row in the dataset. If None,
            it is assumed that the `cv_column_name` in `df_train` is used, by default None.

        Raises
        ------
        ValueError
            If any of the required parameters are missing or if the datasets do not contain the specified columns.

        Examples
        --------
        >>> df_train = pl.from_pandas(
        ...     pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6], "target": [0, 1, 1]})
        ... ).lazy()
        >>> df_val = pl.from_pandas(
        ...     pd.DataFrame({"A": [2, 3, 4], "B": [5, 6, 7], "target": [1, 0, 1]})
        ... ).lazy()
        >>> ua = UnivariateAnalysis(
        ...     "MyModel", df_train, df_val, "target", ["A", "B"], False
        ... )
        """
        self.model_name = model_name
        self.df = to_pl_lf(df_train)
        self.df_val = to_pl_lf(df_val)
        self.target_column_name = target_column_name
        self.feature_column_names = feature_column_names
        self.cv_column_name = cv_column_name if cv_column_name is not None else "cv"
        self.cv_folds = (
            self.df.select([pl.col(self.cv_column_name)]).collect().to_series()
            if cv_folds is None
            else cv_folds
        )
        self.time_series_validation = time_series_validation

        feature_list = []
        transformed_cols = []
        for col in tqdm(
            self.feature_column_names,
            f"Performing univariate analysis on {len(self.feature_column_names)} "
            "features",
        ):
            obj_name = (
                fmt_col_name(col) if not hasattr(self, fmt_col_name(col)) else col
            )
            _column = (
                self.df.select(pl.col(col)).collect().to_series()
                if isinstance(self.df, pl.LazyFrame)
                else self.df[col]
            )
            col_type = get_column_dtype(_column)
            if col_type == "continuous":
                setattr(
                    self,
                    obj_name,
                    Univariate(
                        self.df.filter(pl.col(col).is_not_null()).filter(
                            pl.col(col).is_finite()
                        ),
                        self.df_val.filter(pl.col(col).is_not_null()).filter(
                            pl.col(col).is_finite()
                        ),
                        self.cv_column_name,
                        col,
                        self.target_column_name,
                        time_series_validation=self.time_series_validation,
                    ),
                )
            elif col_type in ("categorical", "binary"):
                setattr(
                    self,
                    obj_name,
                    Univariate(
                        self.df.filter(pl.col(col).is_not_null()),
                        self.df_val.filter(pl.col(col).is_not_null()),
                        self.cv_column_name,
                        col,
                        self.target_column_name,
                        time_series_validation=self.time_series_validation,
                    ),
                )
            elif col_type in ["datetime", "date", "temporal"]:
                pass
            else:
                raise ValueError(
                    f"Column type {col_type} is not supported for univariate analysis"
                )
            feature_list.append(obj_name)

            # Check the skewness of the feature
            skewness = (
                self.df.select(
                    [
                        pl.when(pl.col(col).skew().is_not_null())
                        .then(pl.col(col).skew())
                        .otherwise(0.0)
                        .name.keep()
                    ]
                )
                .collect()
                .item()
            )

            if skewness > 0.5:
                # Considered to be right-skewed, so add a log-transformed
                # version of the feature
                print(f"Feature {col} is right-skewed: skewness = {skewness}")  # noqa: T201
                self.df = self.df.with_columns(
                    [
                        pl.col(col)
                        .log1p()
                        .cast(pl.Float64)
                        .alias(f"log1p_{fmt_col_name(col)}")
                    ]
                )
                self.df_val = self.df_val.with_columns(
                    [
                        pl.col(col)
                        .log1p()
                        .cast(pl.Float64)
                        .alias(f"log1p_{fmt_col_name(col)}")
                    ]
                )

                transformed_obj_name = (
                    f"log1p_{fmt_col_name(col)}"
                    if not hasattr(self, f"log1p_{fmt_col_name(col)}")
                    else f"log1p_{col}"
                )
                setattr(
                    self,
                    transformed_obj_name,
                    Univariate(
                        self.df.filter(
                            pl.col(f"log1p_{fmt_col_name(col)}").is_not_null()
                        ).filter(pl.col(f"log1p_{fmt_col_name(col)}").is_finite()),
                        self.df_val.filter(
                            pl.col(f"log1p_{fmt_col_name(col)}").is_not_null()
                        ).filter(pl.col(f"log1p_{fmt_col_name(col)}").is_finite()),
                        self.cv_column_name,
                        f"log1p_{fmt_col_name(col)}",
                        self.target_column_name,
                        time_series_validation=self.time_series_validation,
                    ),
                )
                transformed_cols.append(f"log1p_{fmt_col_name(col)}")

        self._feature_list = feature_list + transformed_cols
        self.feature_column_names = self._feature_list

    def _sort_features_by_ua(
        self, return_pd: bool = False
    ) -> pd.DataFrame | pl.LazyFrame:
        """
        Sorts features based on their average performance metrics in univariate analysis.

        Parameters
        ----------
        return_pd : bool, optional
            If True, returns a pandas DataFrame. Otherwise, returns a Polars LazyFrame.
            The default is False.

        Returns
        -------
        pd.DataFrame | pl.LazyFrame
            Sorted features based on average performance metrics. The format of the return
            value depends on the `return_pd` parameter.

        Raises
        ------
        AttributeError
            If an expected Univariate object is not found for a feature.
        """
        total_df = []

        for col in self.feature_column_names:
            obj_name = fmt_col_name(col)
            if hasattr(self, obj_name):
                try:
                    ua = getattr(self, obj_name)
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
                    )
                except Exception as e:
                    print(f"Error processing results for feature {col}: {e}")  # noqa: T201
            else:
                print(f"Univariate object not found for feature {col}")  # noqa: T201

        if total_df:
            df = pl.concat(total_df).sort("AUC", descending=True)
            if return_pd:
                return df.collect().to_pandas().set_index("Feature")
            else:
                return df
        else:
            raise AttributeError(
                "No valid Univariate analysis results found for any feature."
            )

    def get_features(self) -> list[str]:
        """Return a list of feature names that have been analyzed."""
        return self._feature_list

    def _get_file_stem(
        self, filename: str | None = None, default: str = "Univariate Analysis Report"
    ) -> str:
        """Generate the file stem from a filename.

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

        >>> _get_file_stem("This.Is.A.Report.pdf", None)
        "This.Is.A.Report"
        """
        if filename is not None:
            # Remove the extension from the filename

            # If there is more than one "." in the filename, remove only the last one
            if filename.count(".") > 1:
                file_stem = ".".join(filename.split(".")[:-1])

            # If there is only one "." in the filename, remove it
            elif filename.count(".") == 1:
                file_stem = filename.split(".")[0]

            # If there are no "." in the filename, return the filename
            else:
                file_stem = filename

            return file_stem

        return default

    def _rpt_filename(
        self,
        file_stem: str | None = None,
        start_num: int | None = None,
        end_num: int | None = None,
        default: str = "Univariate Analysis Report",
    ) -> str:
        """Generate the file name from a filename."""
        if file_stem is not None and (start_num is None or end_num is None):
            return file_stem + ".pdf"
        if start_num is not None and end_num is not None:
            return f"{file_stem}_{start_num}_{end_num+1}.pdf"
        else:
            return default

    @staticmethod
    def _build_desc(total_features: int, max_per_file: int) -> str:
        return (
            f"Building {total_features} univariate analysis reports,"
            f"and packaging in increments of {max_per_file}"
        )

    def _segment_features(
        self, features: list[str], max_per_file: int
    ) -> list[Segment]:
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
        ua = getattr(self, fmt_col_name(feature))
        return ua._add_to_report(rpt)  # noqa: SLF001

    def _generate_segment_report(
        self, segment: dict, filestem_: str, margins_: list[float]
    ) -> None:
        """Generate a report for a specific segment of features."""
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
        filename: str | None = None,
        margins: list[float] | None = None,
        max_per_file: int = 25,
    ) -> None:
        """Build a report for the univariate analysis.

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
        # Handle default values for filename and margins
        filestem_ = self._get_file_stem(filename)
        features = (
            self._sort_features_by_ua()
            .select("Feature")
            .collect()
            .to_series()
            .to_list()
        )
        margins_ = margins if margins is not None else [0.5, 0.5, 0.5, 0.5]

        # Generate segments of features for report generation
        segments = segment_features_for_report(features, max_per_file)

        i = 0
        for segment in tqdm(
            segments,
            desc=f"Building Segment {i+1} out of {len(segments)}. This Segment contains {segments[i].n_features} features.",
        ):
            # Setup filename and initial report pages for each segment
            fn = self._rpt_filename(filestem_, segment.start + 1, segment.end)
            rpt = self._rpt_title_page(fn, margins_)
            rpt = self._rpt_overview_page(rpt, segment.start, segment.end)

            # Process each feature within the current segment
            for feature_name in segment.features:
                rpt = getattr(self, fmt_col_name(feature_name))._add_to_report(rpt)  # noqa: SLF001

            # Finalize the current segment's report
            rpt.build()
            i += 1  # noqa: SIM113 (if I do this I can't use i in the tqdm description)

    def _rpt_overview_page(self, rpt: Report, first_idx: int, last_idx: int) -> Report:
        overview_df = self._sort_features_by_ua().slice(
            first_idx, last_idx - first_idx + 1
        )

        formatted_df = overview_df.select(
            [
                pl.col("Feature")
                .str.replace("_", " ")
                .str.replace("_", " ")
                .str.replace("_", " ")
                .str.replace("_", " ")
                .str.replace("_", " ")
                .str.to_titlecase()
                .str.replace("Log", "log")
                .str.replace("1P", "1p")
                .str.replace("log 1p", "log1p")
                .alias("Feature")
            ]
            + [
                _format_values(col).name.keep()
                for col in overview_df.columns
                if col != "Feature"
            ]
        )

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
            .table(formatted_df.collect().to_pandas().set_index("Feature"))
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
        filename: str | None = None,
        margins: list[float] | None = None,
        date_of_report: datetime.datetime = current_date,
    ) -> Report:
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


# get rid of fold-0 in the report
