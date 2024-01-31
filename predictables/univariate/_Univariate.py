from typing import Any, Callable, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import pandas as pd
import polars as pl
from matplotlib.axes import Axes
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from predictables.univariate.src._get_data import _get_data
from predictables.univariate.src.plots import (
    cdf_plot,
    density_plot,
    quintile_lift_plot,
    roc_curve_plot,
)
from predictables.univariate.src.plots.util import plot_label
from predictables.util import get_column_dtype, get_unique, harmonic_mean, to_pd_df
from predictables.util.report import Report

from ._SingleUnivariate import SingleUnivariate


def get_col(self, col: str) -> List[Union[int, float, str]]:
    """
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

    return attributes + [getattr(self, col)] + [sd]


class Univariate(SingleUnivariate):
    results: pd.DataFrame
    target_name: str
    target: Optional[pd.Series]
    y: Optional[pd.Series]
    Y: Optional[pd.Series]

    feature_name: str
    feature: Optional[pd.Series]
    X: Optional[pd.Series]
    x: Optional[pd.Series]

    fold_name: str
    folds: Optional[pd.Series]

    df_all: pd.DataFrame

    pareto_sort_vector: List[float]

    def __init__(
        self,
        df_: Union[pl.LazyFrame, pl.DataFrame, pd.DataFrame],
        fold_col_: str = "cv",
        feature_col_: Optional[str] = None,
        target_col_: Optional[str] = None,
        **kwargs,
    ) -> None:
        df = to_pd_df(df_)
        if feature_col_ is None:
            feature_col_ = df.columns[1]
        if target_col_ is None:
            target_col_ = df.columns[0]

        # Normalize the column if the cross-validated fit is improved
        df[feature_col_] = self.run_normalization(X=df[feature_col_], y=df[target_col_])

        super().__init__(
            df, fold_col=fold_col_, feature_col=feature_col_, target_col=target_col_
        )

        self.results = pd.DataFrame(index=self.unique_folds + ["mean", "std"])
        self.results.index.name = "fold"
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
            self.results[attribute] = (
                get_col(self, attribute) if hasattr(self, attribute) else None
            )

        # ALIASES
        # =======
        # I am going to alias some of the api syntax errors here if they are
        # reasonable guesses. This needs to be as intuitive as possible.
        dfpd: pd.DataFrame = to_pd_df(df)
        self.target_name: str = (
            self.target_col if isinstance(self.target_col, str) else dfpd.columns[0]
        )
        self.target: Optional[pd.Series] = (
            dfpd.loc[:, self.target_name] if dfpd is not None else None
        )
        self.y: Optional[pd.Series] = (
            dfpd.loc[:, self.target_name] if dfpd is not None else None
        )
        self.Y: Optional[pd.Series] = (
            dfpd.loc[:, self.target_name] if dfpd is not None else None
        )

        self.feature_name = (
            self.feature_col if isinstance(self.feature_col, str) else dfpd.columns[1]
        )
        self.feature: Optional[pd.Series] = (
            dfpd.loc[:, self.feature_name] if dfpd is not None else None
        )
        self.X: Optional[pd.Series] = (
            dfpd.loc[:, self.feature_name] if dfpd is not None else None
        )
        self.x: Optional[pd.Series] = (
            dfpd.loc[:, self.feature_name] if dfpd is not None else None
        )

        self.fold_name: str = (
            self.fold_col
            if (hasattr(self, "fold_col") and isinstance(self.fold_col, str))
            else dfpd.columns[2]
        )
        self.folds: Optional[pd.Series[Any]] = (
            dfpd.loc[:, self.fold_name] if dfpd is not None else None
        )
        self.cv: Optional[pd.Series[Any]] = (
            dfpd.loc[:, self.fold_name] if dfpd is not None else None
        )
        self.fold: Optional[pd.Series[Any]] = (
            dfpd.loc[:, self.fold_name] if dfpd is not None else None
        )

        self.df_all = to_pd_df(
            dfpd if self.df_val is None else pd.concat([dfpd, to_pd_df(self.df_val)])
        )

        self.figsize = (7, 7) if "figsize" not in kwargs else kwargs["figsize"]

        # Pareto sort vector
        pareto_sort_mean = [
            self.auc_test,
            self.acc_test,
            self.f1_test,
            self.precision_test,
            self.recall_test,
            self.mcc_test,
        ]

        pareto_sort_sd = [
            pd.Series([v.auc_test for _, v in self.cv_dict.items()]).std(),
            pd.Series([v.acc_test for _, v in self.cv_dict.items()]).std(),
            pd.Series([v.f1_test for _, v in self.cv_dict.items()]).std(),
            pd.Series([v.precision_test for _, v in self.cv_dict.items()]).std(),
            pd.Series([v.recall_test for _, v in self.cv_dict.items()]).std(),
            pd.Series([v.mcc_test for _, v in self.cv_dict.items()]).std(),
        ]

        self.pareto_sort_vector = (
            pd.DataFrame([[m, s] for m, s in zip(pareto_sort_mean, pareto_sort_sd)])
            .apply(harmonic_mean)
            .values
        )

    def run_normalization(
        self,
        X,
        y,
        normalization_methods: Optional[List[Callable]] = None,
        criterion: Optional[str] = None,
    ):
        if normalization_methods is None:
            normalization_methods = [MinMaxScaler, StandardScaler]

        X = X.values.reshape(-1, 1)

        if get_column_dtype(y) in ["categorical", "binary"]:
            model = LogisticRegression(penalty=None, max_iter=1000, n_jobs=-1)
            criterion = "f1" if criterion is None else criterion
        elif get_column_dtype(y) in ["continuous", "float", "int"]:
            model = LinearRegression(n_jobs=-1)
            criterion = (
                "neg_mean_absolute_percentage_error" if criterion is None else criterion
            )
        else:
            raise ValueError(
                f"Target column {y} has an unsupported dtype: {get_column_dtype(y)}."
            )

        method = ["None"]
        method_obj = [None]
        results = []

        # fit unadjusted data
        results.append(
            cross_val_score(
                model,
                X,
                y,
                cv=5,
                scoring=criterion,
            ).mean()
        )

        for m in normalization_methods:
            method.append(m.__name__)
            method_obj.append(m)
            results.append(
                cross_val_score(
                    model,
                    m().fit_transform(X),
                    y,
                    cv=5,
                    scoring=criterion,
                ).mean()
            )

        best = (
            pd.DataFrame({"method": method, "obj": method_obj, "score": results})
            .sort_values(by="score", ascending=False)
            .iloc[0]
        )

        # return a fitted instance of the best method
        return best["obj"].fit_transform(X) if best["obj"] is not None else X

    def _get_folds(self) -> List[Union[int, float, str]]:
        """
        Helper method that returns an ordered list of the unique elements of
        self.df.cv. Used for reference only.
        """
        return get_unique(self.cv)

    def get_data(
        self, element: str = "x", data: str = "train", fold_n: Optional[int] = None
    ) -> List[Union[int, float, str]]:
        """
        Helper function to get the requested data element.

        Parameters
        ----------
        element : str, optional
            What data element to get. Choices are "x", "y", or "fold"
            for X data (features), y data (target), or data from the nth
            cv fold. Note that `n` must be a named cv fold in the data,
            or an error will be raised.
        data : str, optional
            What data to use for the plot. Choices are "train", "Validate",
            "all".
        fold_n : int, optional
            If element is "fold", which fold to get. Must be a named
            cv fold in the data.

        Returns
        -------
        List[Union[int, float, str]]
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

    def _plot_data(self, data: str = "train"):
        """
        Helper function to get the data for plotting.

        Parameters
        ----------
        data : str, optional
            What data to use for the plot. Choices are "train", "Validate",
            "all".

        Returns
        -------
        Tuple[pd.Series, pd.Series, pd.Series]
            The X, y, and cv data.
        """
        if data not in ["train", "Validate", "all"]:
            raise ValueError(
                f"data must be one of 'train', 'Validate', or 'all'. Got {data}."
            )

        # Set the data
        if data == "train":
            df = to_pd_df(self.df)
        elif data == "Validate":
            df = to_pd_df(self.df_val) if self.df_val is not None else to_pd_df(self.df)
        else:
            df = (
                pd.concat([to_pd_df(self.df), to_pd_df(self.df_val.assign(cv=-42))])
                if to_pd_df(self.df_val)
                else to_pd_df(self.df)
            )

        X = df.loc[:, self.feature_name] if df is not None else self.df.iloc[:, 1]
        y = df.loc[:, self.target_name] if df is not None else self.df.iloc[:, 0]
        cv = df.loc[:, self.fold_col] if df is not None else self.df.iloc[:, 2]

        return X, y, cv

    def plot_cdf(
        self,
        data: str = "train",
        ax: Optional[Axes] = None,
        figsize: Optional[Tuple[float, float]] = None,
        **kwargs,
    ):
        """
        Plots the empirical cumulative distribution function for the target variable in total and for each fold.

        Parameters
        ----------
        data : str, optional
            What data to use for the plot. Choices are "train", "Validate", and "all", "fold-n".
        **kwargs
            Additional keyword arguments passed to the plot function.

        """
        X, y, cv = self._plot_data(data)

        # make plot
        if ax is None:
            _, ax1 = plt.subplots(figsize=self.figsize if figsize is None else figsize)
        else:
            ax1 = ax

        ax1 = cdf_plot(
            X,
            y,
            cv,
            self.feature_name,
            ax=ax1,
            figsize=self.figsize if figsize is None else figsize,
            **kwargs,
        )
        return ax1

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
        """
        Plots the ROC curve for the target variable in total and for each fold.
        """
        if ax is None:
            _, ax0 = plt.subplots(figsize=self.figsize if figsize is None else figsize)
        else:
            ax0 = ax

        ax0 = roc_curve_plot(
            self.y if y is None else y,
            self.yhat_train if yhat is None else yhat,
            self.cv if cv is None else cv,
            self.coef if coef is None else coef,
            self.se if se is None else se,
            self.pvalues if pvalues is None else pvalues,
            ax=ax0,
            figsize=self.figsize if figsize is None else figsize,
            **kwargs,
        )
        return ax0

    def plot_density(
        self,
        data: str = "train",
        feature_name: Optional[str] = None,
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
        Plots the density of the feature at each level of the larget variable, both
        in total and for each fold.
        """
        X, y, cv = self._plot_data(data)
        yhat = self.model.predict(X)

        # make plot
        if ax is None:
            _, ax0 = plt.subplots(figsize=self.figsize if figsize is None else figsize)
        else:
            ax0 = ax

        ax0 = quintile_lift_plot(
            X,
            y,
            yhat,
            ax=ax0,
            figsize=self.figsize if figsize is None else figsize,
            **kwargs,
        )
        return ax0

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
        results = self.results.copy()
        col_multi_index = [
            "Fitted Coef.",
            "Fitted p-Value",
            "Fitted Std. Err.",
            "Conf. Int. Lower",
            "Conf. Int. Upper",
            "Train Accuracy",
            "Val Accuracy",
            "Train AUC",
            "Val AUC",
            "Train F1",
            "Test F1",
            "Train Precision",
            "Val Precision",
            "Train Recall",
            "Val Recall",
            "Train MCC",
            "Val MCC",
            "Train Log-Loss",
            "Val Log-Loss",
        ]

        row_multi_index = [f"Fold-{i}" for i in self.unique_folds] + [
            "Agg. Mean",
            "Agg. SD",
        ]

        if use_formatting:
            pct_cols = [
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
            ]

            # Percent-format with single decimal point:
            for col in pct_cols:
                results[col] = results[col].apply(lambda x: f"{x:.1%}")

            # Hierarchy of formatting depending on size of median value in col
            for col in [c for c in results.columns.tolist() if c not in pct_cols]:
                m = results[col].median()
                if m > 1.0:
                    results[col] = results[col].apply(lambda x: f"{x:.2f}")
                elif m > 0.1:
                    results[col] = results[col].apply(lambda x: f"{x:.3f}")
                else:
                    results[col] = results[col].apply(lambda x: f"{x:.1e}")

            # Apply the multi-index at the end
            results.columns = col_multi_index
            results.index = row_multi_index

        return results.T

    

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
                y=self.y,
                yhat=self.yhat_train,
                cv=self.df.cv,
                coef=self.coef,
                se=self.se,
                pvalues=self.pvalues,
                figsize=self.figsize,
            )

        def quintile():
            return self.plot_quintile_lift(data="train", figsize=self.figsize)

        return (
            rpt.h2("Univariate Report")
            .h3(f"{plot_label(self.feature_name, incl_bracket=False)} - Results")
            .spacer(0.5)
            .table(self.get_results())
            .page_break()
            .h2("Univariate Report")
            .h3(
                f"{plot_label(self.feature_name, incl_bracket=False)} - Kernel Density Plot"
            )
            .plot(density)
            .page_break()
            .h2("Univariate Report")
            .h3(
                f"{plot_label(self.feature_name, incl_bracket=False)} - Empirical CDF Plot"
            )
            .plot(cdf)
            .page_break()
            .h2("Univariate Report")
            .h3(f"{plot_label(self.feature_name, incl_bracket=False)} - ROC Curve")
            .plot(roc)
            .page_break()
            .h2("Univariate Report")
            .h3(f"{plot_label(self.feature_name, incl_bracket=False)} - Quintile Lift")
            .plot(quintile)
            .page_break()
        )
