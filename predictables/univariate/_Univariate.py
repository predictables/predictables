from typing import List, Optional, Union

import matplotlib.pyplot as plt
import pandas as pd
import polars as pl
from matplotlib.axes import Axes

from predictables.univariate.src._get_data import _get_data
from predictables.univariate.src.plots import (
    cdf_plot,
    density_plot,
    quintile_lift_plot,
    roc_curve_plot,
)
from predictables.util import get_unique, to_pd_df, to_pl_lf
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

    attributes = [getattr(self.cv[fold], col) for fold in self.unique_folds]
    sd = pd.Series(attributes).std()

    return attributes + [getattr(self, col)] + [sd]


class Univariate(SingleUnivariate):
    def __init__(
        self,
        df: Union[pl.LazyFrame, pl.DataFrame, pd.DataFrame],
        fold_col: str = "cv",
        feature_col: Optional[str] = None,
        target_col: Optional[str] = None,
        **kwargs,
    ):
        df = to_pl_lf(df)
        if feature_col is None:
            feature_col = df.columns[1]
        if target_col is None:
            target_col = df.columns[0]

        super().__init__(
            df, fold_col=fold_col, feature_col=feature_col, target_col=target_col
        )
        self.results = pd.DataFrame(
            index=sorted(self.unique_folds.tolist()) + ["mean", "std"]
        )
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
            "auc_train",
            "auc_test",
        ]:
            self.results[attribute] = get_col(self, attribute)

        # ALIASES
        # =======
        # I am going to alias some of the api syntax errors here if they are
        # reasonable guesses. This needs to be as intuitive as possible.
        self.target_name = (
            self.target_col if isinstance(self.target_col, str) else self.df.columns[0]
        )
        self.target = (
            to_pd_df(self.df).loc[:, self.target_name] if self.df is not None else None
        )
        self.y = (
            to_pd_df(self.df).loc[:, self.target_name] if self.df is not None else None
        )
        self.Y = (
            to_pd_df(self.df).loc[:, self.target_name] if self.df is not None else None
        )

        self.feature_name = (
            self.feature_col
            if isinstance(self.feature_col, str)
            else self.df.columns[1]
        )
        self.feature = (
            to_pd_df(self.df).loc[:, self.feature_name] if self.df is not None else None
        )
        self.X = (
            to_pd_df(self.df).loc[:, self.feature_name] if self.df is not None else None
        )
        self.x = (
            to_pd_df(self.df).loc[:, self.feature_name] if self.df is not None else None
        )

        self.fold_name = (
            self.fold_col if isinstance(self.fold_col, str) else self.df.columns[2]
        )
        self.folds = (
            to_pd_df(self.df).loc[:, self.fold_name] if self.df is not None else None
        )
        self.cv = (
            to_pd_df(self.df).loc[:, self.fold_name] if self.df is not None else None
        )
        self.fold = (
            to_pd_df(self.df).loc[:, self.fold_name] if self.df is not None else None
        )

        self.df_all = to_pd_df(
            self.df if self.df_val is None else pd.concat([self.df, self.df_val])
        )

        self.figsize = (6, 6) if "figsize" not in kwargs else kwargs["figsize"]

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
            What data to use for the plot. Choices are "train", "test",
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
        if data not in ["train", "test", "all"]:
            raise ValueError(
                f"data must be one of 'train', 'test', or 'all'. Got {data}."
            )

        # Set the data
        if data == "train":
            df: pd.DataFrame = to_pd_df(self.df)
        elif data == "test":
            df: pd.DataFrame = (
                to_pd_df(self.df_val) if self.df_val is not None else to_pd_df(self.df)
            )
        else:
            df: pd.DataFrame = (
                pd.concat([to_pd_df(self.df), to_pd_df(self.df_val.assign(cv=-42))])
                if to_pd_df(self.df_val)
                else to_pd_df(self.df)
            )

        X: Optional[pd.DataFrame] = (
            df.loc[:, self.feature_name] if df is not None else None
        )
        y: Optional[pd.DataFrame] = (
            df.loc[:, self.target_name] if df is not None else None
        )
        cv: Optional[pd.DataFrame] = (
            df.loc[:, self.fold_col] if df is not None else None
        )

        return X, y, cv

    def plot_cdf(self, data: str = "train", **kwargs):
        """
        Plots the empirical cumulative distribution function for the target variable in total and for each fold.

        Parameters
        ----------
        data : str, optional
            What data to use for the plot. Choices are "train", "test", and "all", "fold-n".
        **kwargs
            Additional keyword arguments passed to the plot function.

        """
        X, y, cv = self._plot_data(data)

        # make plot
        _, ax = plt.subplots(
            figsize=self.figsize if "figsize" not in kwargs else kwargs["figsize"]
        )
        ax = cdf_plot(
            X,
            y,
            cv,
            self.feature_name,
            ax=ax,
            figsize=self.figsize if "figsize" not in kwargs else kwargs["figsize"],
            **kwargs,
        )
        return ax

    def plot_roc_curve(
        self,
        y: Optional[Union[pd.Series, pl.Series]] = None,
        yhat: Optional[Union[pd.Series, pl.Series]] = None,
        cv: Optional[Union[pd.Series, pl.Series]] = None,
        coef: Optional[float] = None,
        se: Optional[float] = None,
        pvalues: Optional[float] = None,
        ax: Optional[Axes] = None,
        **kwargs,
    ) -> Axes:
        """
        Plots the ROC curve for the target variable in total and for each fold.
        """
        if ax is None:
            _, ax0 = plt.subplots(
                figsize=self.figsize if "figsize" not in kwargs else kwargs["figsize"]
            )
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
            figsize=self.figsize if "figsize" not in kwargs else kwargs["figsize"],
            **kwargs,
        )
        return ax0

    def plot_density(
        self,
        data: str = "train",
        feature_name: Optional[str] = None,
        ax: Optional[Axes] = None,
        **kwargs,
    ) -> Axes:
        """
        Plots the density of the feature at each level of the larget variable, both
        in total and for each fold.
        """
        X, y, cv = self._plot_data(data)

        # make plot
        if ax is None:
            _, ax0 = plt.subplots(
                figsize=self.figsize if "figsize" not in kwargs else kwargs["figsize"]
            )
        else:
            ax0 = ax

        ax0 = density_plot(
            X,
            y,
            cv,
            feature_name if feature_name is not None else self.feature_name,
            ax=ax0,
            figsize=self.figsize if "figsize" not in kwargs else kwargs["figsize"],
            **kwargs,
        )
        return ax0
