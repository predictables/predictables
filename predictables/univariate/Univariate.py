from typing import List, Union

import matplotlib.pyplot as plt
import pandas as pd
import polars as pl

from predictables.univariate.src._get_data import _get_data
from predictables.univariate.src.plots import (
    cdf_plot,
    density_plot,
    quintile_lift_plot,
    roc_curve_plot,
)
from predictables.util import get_unique

from .SingleUnivariate import SingleUnivariate


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

    out = attributes + [getattr(self, col)] + [sd]
    return out


class Univariate(SingleUnivariate):
    def __init__(
        self,
        df: pl.LazyFrame,
        fold_col: str = "cv",
        feature_col: str = None,
        target_col: str = None,
        **kwargs,
    ):
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
        self.target_name = self.target_col
        self.target = self.df[self.target_col]
        self.y = self.df[self.target_col]
        self.Y = self.df[self.target_col]

        self.feature_name = self.feature_col
        self.feature = self.df[self.feature_col]
        self.X = self.df[self.feature_col]
        self.x = self.df[self.feature_col]

        self.folds = self.df[self.fold_col]
        self.cv = self.df[self.fold_col]
        self.fold = self.df[self.fold_col]

        self.df_all = (
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
        self, element: str = "x", data: str = "train", fold_n: int = None
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

    def plot_cdf(self, data: str = "train"):
        """
        Plots the empirical cumulative distribution function for the target variable in total and for each fold.

        Parameters
        ----------
        data : str, optional
            What data to use for the plot. Choices are "train", "test", and "all", "fold-n".
        """
        if data not in ["train", "test", "all"]:
            raise ValueError(
                f"data must be one of 'train', 'test', or 'all'. Got {data}."
            )

        # Set the data
        if data == "train":
            df = self.df
        elif data == "test":
            df = self.df_val
        else:
            df = pd.concat([self.df, self.df_val.assign(cv=-42)])

        X = df[self.feature_name]
        y = df[self.target_name]
        cv = df[self.fold_col]

        # make plot
        fig, ax = plt.subplots(figsize=self.figsize)
        ax = cdf_plot(X, y, cv, self.feature_name, ax=ax)
        return ax
