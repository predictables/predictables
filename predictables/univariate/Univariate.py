from typing import List, Union

import pandas as pd
import polars as pl

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
        The name of the column to get. Choices are "coef", "pvalues", "se", "lower_ci", "upper_ci", "acc_train", "acc_test", "auc_train", "auc_test", "f1_train", "f1_test", "precision_train", "precision_test", "recall_train", "recall_test", "mcc_train", "mcc_test", "logloss_train", "logloss_test", "auc_train", "auc_test".

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

    def _get_folds(self) -> List[Union[int, float, str]]:
        """
        Helper method that returns an ordered list of the unique elements of
        self.df.cv. Used for reference only.
        """
        return get_unique(self.cv)

    def _get_data(
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
        element = element.lower()
        data = data.lower()

        if data not in ["train", "test", "all"]:
            raise ValueError(
                f"data must be one of 'train', 'test', or 'all'. Got {data}."
            )
        if element not in ["x", "y", "fold"]:
            raise ValueError(
                f"element must be one of 'x', 'y', or 'fold'. Got {element}."
            )
        if element == "fold" and fold_n not in self._get_folds():
            raise ValueError(
                f"fold_n must be one of {self._get_folds()}. Got {fold_n}."
            )

        # Extract the fold number if provided
        if element.startswith("fold-"):
            fold = int(element.split("-")[1])
            element = "fold"
        else:
            fold = None

        def row_filter(data, element, fold=None):
            if data == "all":
                return self.df_all
            elif data == "train":
                return self.df.loc[self.fold.eq(fold)] if fold is not None else self.df
            elif data == "test":
                if fold is not None:
                    return self.df.loc[self.fold.ne(fold)]
                else:
                    return self.df_val if self.df_val is not None else self.df
            else:
                raise ValueError(
                    f"data must be one of 'train', 'test', or 'all'. Got {data}."
                )

        def col_filter(data, element, fold=None):
            if element == "x":
                return row_filter(data, element, fold).loc[:, self.feature_col]
            elif element == "y":
                return row_filter(data, element, fold).loc[:, self.target_col]
            elif element == "fold":
                return row_filter(data, element, fold).loc[:, self.fold_col]
            else:
                raise ValueError(
                    f"element must be one of 'x', 'y', or 'fold'. Got {element}."
                )

        return col_filter(data, element, fold)

    def plot_cdf(self, data: str = "train"):
        """
        Plots the empirical cumulative distribution function for the target variable in total and for each fold.

        Parameters
        ----------
        data : str, optional
            What data to use for the plot. Choices are "train", "test", and "all", "fold-n".
        """
        # Set the data
        ax = cdf_plot(x=self.X)
