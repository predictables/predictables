from __future__ import annotations


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
from matplotlib.axes import Axes
from sklearn.preprocessing import MinMaxScaler, StandardScaler  # type: ignore

from predictables.univariate._BaseModel import Model
from predictables.univariate.src._get_data import _get_data
from predictables.univariate.src.plots import (
    cdf_plot,
    density_plot,
    quintile_lift_plot,
    roc_curve_plot,
)
from predictables.univariate.src.plots.util import plot_label
from predictables.util import (
    DebugLogger,
    col_name_for_report,
    get_unique,
    to_pd_df,
    to_pd_s,
)
from predictables.util.report import Report

dbg = DebugLogger(working_file="_Univariate.py")


def get_col(self: Univariate, col: str) -> list[int | float | str]:
    """Get the requested column from the data.

    Helper function to get the requested column from the data.

    Parameters
    ----------
    col : str
        The name of the column to get. Choices are
            - "coef"
            - "pvalues",
            - "se"
            - "lower_ci"
            - "upper_ci"
            - "acc_train"
            - "acc_test"
            - "auc_train"
            - "auc_test"
            - "f1_train"
            - "f1_test"
            - "precision_train"
            - "precision_test"
            - "recall_train"
            - "recall_test"
            - "mcc_train"
            - "mcc_test"
            - "logloss_train"
            - "logloss_test"
            - "auc_train"
            - "auc_test"

    Returns
    -------
    List[Union[int, float, str]]
        The values for the requested column.

    Examples
    --------
    # Assume you have fit a model with coefficients [0.1, 0.2, 0.3, 0.4, 0.5]
    # and a standard deviation of 0.01.

    >>> get_col(self, "coef")
    [0.1, 0.2, 0.3, 0.4, 0.5]

    >>> get_col(self, "std")
    [0.01, 0.01, 0.01, 0.01, 0.01]
    """
    attributes = [getattr(self.cv_dict[fold], col) for fold in self.unique_folds]
    sd = pd.Series(attributes).std()

    return [*attributes, getattr(self, col), sd]


class Univariate(Model):
    target_name: str
    target: pd.Series | None
    y: pd.Series | None
    Y: pd.Series | None

    feature_name: str
    feature: pd.Series | None
    X: pd.Series | None
    x: pd.Series | None

    fold_name: str
    folds: pd.Series | None

    normalization_obj: Optional[Union[MinMaxScaler, StandardScaler]]

    def __init__(
        self,
        df_: Union[pl.LazyFrame, pl.DataFrame, pd.DataFrame],
        df_val_: Union[pl.LazyFrame, pl.DataFrame, pd.DataFrame],
        fold_col_: str = "cv",
        feature_col_: Optional[str] = None,
        target_col_: Optional[str] = None,
        time_series_validation: bool = True,
        skewness_threshold: float = 0.5,
        **kwargs,
    ) -> None:
        df: pd.DataFrame = to_pd_df(df_)
        df_val: pd.DataFrame = to_pd_df(df_val_)
        if feature_col_ is None:
            feature_col_ = df.columns[1]
        if target_col_ is None:
            target_col_ = df.columns[0]

        super().__init__(
            df,
            df_val,
            fold_col=fold_col_,
            feature_col=feature_col_,
            target_col=target_col_,
            time_series_validation=time_series_validation,
        )

        self.skewness_threshold = skewness_threshold

        # Normalize the column if the cross-validated fit is improved
        df[feature_col_] = self.standardize(X=df[[feature_col_]])
        self.unique_folds = get_unique(
            self.df.select(self.fold_col).collect().to_pandas()[self.fold_col]
        )
        self.unique_folds_str = [f"Fold-{f}".zfill(2) for f in self.unique_folds]

        self.cv_dict = {}
        for fold in self.unique_folds:
            self.cv_dict[fold] = Model(
                self.df,
                fold_n=fold,
                fold_col=self.fold_col,
                feature_col=(
                    self.feature_col if self.feature_col is not None else None
                ),
                target_col=(self.target_col if self.target_col is not None else None),
            )

        self.agg_results = pl.from_pandas(
            pd.DataFrame({"fold": [*self.unique_folds_str, "mean", "std"]})
        ).lazy()
        for attribute in [
            "coef",
            "pvalues",
            "se",
            "lower_ci",
            "upper_ci",
            "acc_train",
            "acc_test",
            "auc_train",
            "auc_test",
            "f1_train",
            "f1_test",
            "precision_train",
            "precision_test",
            "recall_train",
            "recall_test",
            "mcc_train",
            "mcc_test",
            "logloss_train",
            "logloss_test",
        ]:
            if (self.results is not None) and (attribute in self.results.columns):
                # Get the attribute from the cv_dict for each fold
                att = [
                    self.cv_dict[fold].results.select(attribute).collect().item(0, 0)
                    for fold in self.unique_folds
                ]

                # Get the mean and standard deviation of the attribute
                std = pd.Series(att).std()
                att += [self.results.select(pl.col(attribute)).collect().item(0, 0)]
                att += [std]
                self.agg_results = self.agg_results.with_columns(
                    pl.Series(att).alias(attribute)
                )

            else:
                dbg.msg(
                    f"[{self.feature_col}]: Attribute {attribute} not found in "
                    "self | Ux0001g"
                )

        # ALIASES
        # =======
        # I am going to alias some of the api syntax errors here if they are
        # reasonable guesses. This needs to be as intuitive as possible.
        dfpd: pd.DataFrame = to_pd_df(df)
        self.target_name: str = (
            self.target_col if isinstance(self.target_col, str) else dfpd.columns[0]
        )
        self.target: pd.Series | None = (
            dfpd.loc[:, self.target_name] if dfpd is not None else None
        )
        self.y: pd.Series | None = (
            dfpd.loc[:, self.target_name] if dfpd is not None else None
        )
        self.Y: pd.Series | None = (
            dfpd.loc[:, self.target_name] if dfpd is not None else None
        )

        self.feature_name = (
            self.feature_col if isinstance(self.feature_col, str) else dfpd.columns[1]
        )
        self.feature: pd.Series | None = (
            dfpd.loc[:, self.feature_name] if dfpd is not None else None
        )
        self.X: pd.Series | None = (
            dfpd.loc[:, self.feature_name] if dfpd is not None else None
        )
        self.x: pd.Series | None = (
            dfpd.loc[:, self.feature_name] if dfpd is not None else None
        )

        self.fold_name: str = (
            self.fold_col
            if (hasattr(self, "fold_col") and isinstance(self.fold_col, str))
            else dfpd.columns[2]
        )
        self.folds: pd.Series | None = (
            dfpd.loc[:, self.fold_name] if dfpd is not None else None
        )
        self.cv: pd.Series | None = (
            dfpd.loc[:, self.fold_name] if dfpd is not None else None
        )
        self.fold: pd.Series | None = (
            dfpd.loc[:, self.fold_name] if dfpd is not None else None
        )

        self.figsize = kwargs.get("figsize", (7, 7))

        self.skewness = (
            self.df.select(pl.col(self.feature_name).skew().name.keep())
            .collect()
            .item()
        )
        self.is_right_skewed = self.skewness > self.skewness_threshold
        self.is_left_skewed = self.skewness < -self.skewness_threshold

    def _get_folds(self) -> list[int | float | str]:
        """Get an ordered list of the unique elements of self.df.cv.

        Helper method that returns an ordered list of the unique elements of
        self.df.cv. Used for reference only.
        """
        return get_unique(
            to_pd_s(self.cv)
            if isinstance(self.cv, (pl.Series, pd.Series))
            else self.df.select(self.fold_col).collect().to_pandas()[self.fold_col]
        )

    def get_data(
        self, element: str = "x", data: str = "train", fold_n: int | None = None
    ) -> list[int | float | str]:
        """Get the requested data element.

        This is a helper function to get the requested data element.

        Parameters
        ----------
        element : str, optional
            What data element to get. Choices are "x", "y", or "fold"
            for X data (features), y data (target), or data from the nth
            cv fold. Note that `n` must be a named cv fold in the data,
            or an error will be raised.
        data : str, optional
            What data to use for the plot. Choices are "train", "test",
            "all".
        fold_n : int, optional
            If element is "fold", which fold to get. Must be a named
            cv fold in the data.

        Returns
        -------
        list[int | float | str]
            The values for the requested column.
        """
        return _get_data(
            self.df,
            self.df_val,
            element,
            data,
            fold_n,
            self.feature_name,
            self.target_name,
            "cv",
        )

    def _plot_data(self, data: str = "train") -> tuple[pd.Series, pd.Series, pd.Series]:
        """Get the data for the plot.

        This is a helper function to get the data for plotting.

        Parameters
        ----------
        data : str, optional
            What data to use for the plot. Choices are "train", "test",
            "all".

        Returns
        -------
        tuple[pd.Series, pd.Series, pd.Series]
            The X, y, and cv data.
        """
        if data not in ["train", "test", "all"]:
            raise ValueError(
                f"data must be one of 'train', 'test', or 'all'. Got {data}."
            )

        # Set the data
        if data == "train":
            df = to_pd_df(self.df)
            cv = (
                df.loc[:, self.fold_col]
                if df is not None
                else self.df.select(pl.col(self.fold_col))
                .collect()
                .to_pandas()[self.fold_col]
            )
        elif data == "test":
            df = to_pd_df(self.df_val) if self.df_val is not None else to_pd_df(self.df)
            df = df.assign(cv=-42)
            cv = df["cv"]
        else:
            df = (
                pd.concat(
                    [
                        to_pd_df(self.df),
                        to_pd_df(self.df_val.with_columns(pl.lit(-42).alias("cv"))),
                    ]
                )
                if to_pd_df(self.df_val)
                else to_pd_df(self.df)
            )
            cv = df["cv"]

        X = (  # noqa: N806 (X is traditional for features in ML)
            df.loc[:, self.feature_name]
            if df is not None
            else self.df.select(self.feature_name)
        )
        y = (
            df.loc[:, self.target_name]
            if df is not None
            else self.df.select(self.target_name)
        )

        return X, y, cv

    def plot_cdf(
        self,
        data: str = "train",
        ax: plt.Axes | None = None,
        figsize: tuple[float, float] | None = None,
        **kwargs,
    ) -> plt.Axes:
        """Plot the empirical cumulative distribution function.

        Plots the empirical CDF for the target variable in total and for
        each fold.

        Parameters
        ----------
        data : str, optional
            What data to use for the plot. Choices are "train", "test", and
            "all", "fold-n".
        **kwargs
            Additional keyword arguments passed to the plot function.

        """
        X, y, cv = self._plot_data(data)

        # make plot
        if ax is None:
            _, ax1 = plt.subplots(figsize=self.figsize if figsize is None else figsize)
        else:
            ax1 = ax

        return cdf_plot(
            X,
            y,
            cv,
            self.feature_name,
            ax=ax1,
            figsize=self.figsize if figsize is None else figsize,
            **kwargs,
        )

    def plot_roc_curve(
        self,
        y: Optional[Union[pd.Series, pl.Series]] = None,
        yhat: Optional[Union[pd.Series, pl.Series]] = None,
        cv: Optional[Union[pd.Series, pl.Series]] = None,
        coef: Optional[float] = None,
        se: Optional[float] = None,
        pvalues: Optional[float] = None,
        ax: Optional[Axes] = None,
        figsize: Optional[Tuple[float, float]] = None,
        **kwargs,
    ) -> Axes:
        """Plot the ROC curve for the target variable in total and for each fold."""
        if ax is None:
            _, ax0 = plt.subplots(figsize=self.figsize if figsize is None else figsize)
        else:
            ax0 = ax

        if cv is None:
            cv = (
                to_pd_s(self.cv)
                if self.cv is not None
                else to_pd_s(self.df.select(self.fold_col).collect()[self.fold_col])
            )
        else:
            cv = to_pd_s(cv)

        ax0 = roc_curve_plot(
            to_pd_s(self.GetY("train")) if y is None else to_pd_s(y),
            to_pd_s(self.GetYhat("train")) if yhat is None else to_pd_s(yhat),
            cv,
            self.time_series_validation,
            (
                to_pd_df(self.agg_results).loc["Ave.", "coef"].to_numpy()  # type: ignore
                if coef is None
                else coef
            ),
            (
                to_pd_df(self.agg_results).loc["Ave.", "coef"].to_numpy()  # type: ignore
                if se is None
                else se
            ),
            self.get("pvalues") if pvalues is None else pvalues,
            ax=ax0,
            figsize=self.figsize if figsize is None else figsize,
            **kwargs,
        )
        return ax0

    def plot_density(
        self,
        data: str = "train",
        ax: Optional[Axes] = None,
        figsize: Optional[Tuple[float, float]] = None,
        **kwargs,
    ) -> Axes:
        """
        Plots the density of the feature at each level of the larget variable, both
        in total and for each fold.
        """
        X, y, cv = self._plot_data(data)

        # make plot
        if ax is None:
            _, ax0 = plt.subplots(figsize=self.figsize if figsize is None else figsize)
        else:
            ax0 = ax

        ax0 = density_plot(
            X,
            y,
            cv,
            X.min(),
            X.max(),
            ax=ax0,
            figsize=self.figsize if figsize is None else figsize,
            **kwargs,
        )
        return ax0

    def plot_quintile_lift(
        self,
        data: str = "train",
        ax: Optional[Axes] = None,
        figsize: Optional[Tuple[float, float]] = None,
        **kwargs,
    ) -> Axes:
        """
        Plots the quintile lift for the target variable in total and for each fold.
        Quintile lift is a grouped bar plot with each quintile of the feature on the
        x-axis and the mean target value for each quintile on the y-axis. There are
        bars for both the actual target value and the predicted target value.

        This plot is useful for understanding how well the model is able to segment
        the target variable based on the feature variable, and in the specific context
        of a univariate analysis, how well the feature variable is able to predict the
        target variable by itself, eg with no intercept, no other features, no
        regularization, etc.

        Parameters
        ----------
        data : str, optional
            What data to use for the plot. Choices are "train", "test", and
            "all", "fold-n".
        ax : Axes, optional
            The axes to plot on, by default None. If None, a new figure and axes
            will be created.
        figsize : Tuple[float, float], optional
            The size of the figure, by default None. If None, the default figure
            size will be used.
        **kwargs
            Additional keyword arguments passed to the plot function.

        Returns
        -------
        Axes
            The axes object for the plot.

        Examples
        --------
        # Plot the quintile lift for the training data in a univariate analysis `uni`
        # with the default figure size.
        >>> uni.plot_quintile_lift(data="train")

        # Plot the quintile lift for the test data in a univariate analysis `uni`
        # with a figure size of (10, 10).
        >>> uni.plot_quintile_lift(data="test", figsize=(10, 10))

        # Plot the quintile lift for the training data in a univariate analysis `uni`
        # with the default figure size and a custom color for the predicted target
        # value.
        >>> uni.plot_quintile_lift(data="train", color="red")

        # Plot the quintile lift for the training data in a univariate analysis `uni`
        # with the default figure size and a custom title.
        >>> uni.plot_quintile_lift(data="train", title="Quintile Lift Plot")
        """
        X, y, _ = self._plot_data(data)
        yhat = self.predict(X)

        # make plot
        if ax is None:
            _, ax0 = plt.subplots(figsize=self.figsize if figsize is None else figsize)
        else:
            ax0 = ax

        yhat_polars = pl.Series(yhat)
        ax0 = quintile_lift_plot(
            X,
            y,
            yhat_polars,
            ax=ax0,
            figsize=self.figsize if figsize is None else figsize,
            **kwargs,
        )
        return ax0

    def _add_to_report(self, rpt: Optional[Report] = None, **kwargs):
        if rpt is None:
            rpt = Report(**kwargs)

        def density():
            return self.plot_density(
                data="train", feature_name=self.feature_name, figsize=self.figsize
            )

        def cdf():
            return self.plot_cdf(data="train", figsize=self.figsize)

        def roc():
            return self.plot_roc_curve(
                y=self.GetY("train"),
                yhat=self.GetYhat("train"),
                cv=self.df.select(self.fold_col).collect().to_pandas()[self.fold_col],  # type: ignore
                time_series_validation=self.time_series_validation,
                coef=self.get("coef"),
                se=self.get("se"),
                pvalues=self.get("pvalues"),
                figsize=self.figsize,
            )

        def quintile():
            return self.plot_quintile_lift(data="test", figsize=self.figsize)

        rpt = (
            rpt.h2("Univariate Report")
            .h3(f"{plot_label(self.feature_name, incl_bracket=False)} - Results")
            .spacer(0.5)
            .table(self.get_results())
            .page_break()
        )

        try:
            rpt = (
                rpt.h2("Univariate Report")
                .h3(
                    f"{plot_label(self.feature_name, incl_bracket=False)} - Kernel Density Plot"
                )
                .plot(density)
                .spacer(0.125)
                .caption(
                    "This plot shows the Gaussian kernel density for each level of the "
                    "target variable, both in total and for each fold. "
                    "The x-axis represents the feature variable, and the y-axis represents "
                    "the density of the target variable. "
                    "The cross-validation folds are included in slightly washed-out colors "
                    "to help understand the variability of the data. "
                    "There are annotations with the results of a t-test for the difference "
                    "in means between the feature variable at each level of the target "
                    "variable. "
                    "The annotations corresponding to the color of the target variable "
                    "level show the mean/median ratio to help understand differences in "
                    "skewness between the levels of the target variable."
                )
                # TODO: Add in a table for the t-test results (Issue #62 on GitHub)
                .page_break()
            )
        except np.linalg.LinAlgError as err:
            rpt = (
                rpt.h2("Univariate Report")
                .h3(
                    f"{plot_label(self.feature_name, incl_bracket=False)} - Kernel Density Plot"
                )
                .p(
                    "The kernel density plot could not be created due to a "
                    "LinAlgError. This is likely due to the feature variable "
                    "being constant, and the kernel density plot is not "
                    "informative.\n"
                    "This is likely a good indication that this variable is not "
                    "useful for predicting the target variable.\n"
                )
                .p(f"Error message:\n{err}")
                .page_break()
            )

        return (
            rpt.h2("Univariate Report")
            .h3(
                f"{plot_label(self.feature_name, incl_bracket=False)} "
                "- Empirical CDF Plot"
            )
            .plot(cdf)
            .spacer(0.125)
            .caption(
                "This plot shows the empirical cumulative distribution function for "
                "each level of the target variable, both in total and for each fold. "
                "The x-axis represents the feature variable, and the y-axis represents "
                "the cumulative distribution of the target variable. "
                "The cross-validation folds are included in slightly washed-out colors "
                "to help understand the variability of the data, and whether or not it "
                "is reasonable to assume that the data is drawn from different "
                "distributions."
            )
            .page_break()
            .h2("Univariate Report")
            .h3(f"{plot_label(self.feature_name, incl_bracket=False)} - ROC Curve")
            .plot(roc)
            .spacer(0.125)
            .caption(
                "This plot shows the receiver operating characteristic (ROC) curve "
                "for the target variable in total and for each fold. "
                "The x-axis represents the false positive rate, and the y-axis "
                "represents the true positive rate. This is based on a simple "
                "Logistic Regression model with no regularization, no intercept, "
                "and no other features. "
                "Annotations are on the plot to help understand the results of the "
                "model, including the coefficient, standard error, and p-value for "
                "the feature variable. "
                "The cross-validation folds are used to create the grey region around "
                "the mean ROC curve to help understand the variability of the data."
            )
            .caption(
                "Significance of the ROC curve is determined based on a modified "
                "version the method from DeLong et al. (1988). In brief, the AUC "
                "is assumed to be normally distributed, and I calculate the empirical "
                "standard error from the cross-validated AUC values. "
                "I then calculate a z-score for the AUC, and use the z-score to "
                "calculate a p-value. "
                "The p-value is then used to determine the significance of the AUC. "
                "This is a simple test, and should be used with caution."
            )
            .page_break()
            .h2("Univariate Report")
            .h3(f"{plot_label(self.feature_name, incl_bracket=False)} - Quintile Lift")
            .plot(quintile)
            .spacer(0.125)
            .caption(
                "The quintile lift plot is meant to show the power of the single "
                "feature to discriminate between the highest and lowest quintiles "
                "of the target variable."
            )
            .page_break()
        )

    def get_results(self, use_formatting: bool = True) -> pd.DataFrame:
        """
        Returns the self.results attribute, formatted for the univariate report.

        Parameters
        ----------
        use_formatting : bool, optional
            Whether to use formatting or not. Defaults to True. If set to False,
            the unformated results will be returned as a pandas DataFrame.

        Returns
        -------
        pd.DataFrame
            The results dataframe.
        """
        results = to_pd_df(
            self.agg_results.select(
                [
                    pl.col("fold"),
                    pl.col("coef"),
                    pl.col("pvalues"),
                    pl.col("se"),
                    pl.col("lower_ci"),
                    pl.col("upper_ci"),
                    pl.col("acc_test"),
                    pl.col("auc_test"),
                    pl.col("f1_test"),
                    pl.col("precision_test"),
                    pl.col("recall_test"),
                    pl.col("mcc_test"),
                ]
            )
        )
        if use_formatting:
            pct_cols = [
                # "acc_train",
                "acc_test",
                # "auc_train",
                "auc_test",
                # "f1_train",
                "f1_test",
                # "precision_train",
                "precision_test",
                # "recall_train",
                "recall_test",
                # "mcc_train",
                "mcc_test",
            ]

            # Percent-format with single decimal point:
            for col in pct_cols:
                results[col] = results[col].apply(lambda x: f"{x:.1%}")

            # Hierarchy of formatting depending on size of median value in col
            for col in [c for c in results.columns.tolist()[1:] if c not in pct_cols]:
                m = results[col].median()
                if np.abs(m) > 1.0:
                    results[col] = results[col].apply(lambda x: f"{x:.2f}")
                elif np.abs(m) > 0.1:
                    results[col] = results[col].apply(lambda x: f"{x:.3f}")
                else:
                    results[col] = results[col].apply(lambda x: f"{x:.1e}")

            # Apply an index at the end
            results.columns = pd.Index(
                [
                    (
                        col_name_for_report(c).replace(" ", "\n")
                        if c != "fold"
                        else "CV\nFold"
                    )
                    for c in results.columns.tolist()
                ]
            )

        return results.set_index("CV\nFold")
