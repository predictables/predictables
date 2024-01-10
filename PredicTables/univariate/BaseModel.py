from typing import Union

import numpy as np
import pandas as pd
import polars as pl
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import precision_recall_curve, roc_curve

from PredicTables.util import get_column_dtype

from .src import (
    fit_sk_linear_regression,
    fit_sk_logistic_regression,
    fit_sm_linear_regression,
    fit_sm_logistic_regression,
    time_series_validation_filter,
)


class Model:
    """
    A class to fit a model to a dataset. Used in the univariate analysis to fit a simple model to each variable.
    """

    def __init__(
        self,
        df: Union[pd.DataFrame, pl.DataFrame, pl.LazyFrame],
        df_val: Union[pd.DataFrame, pl.DataFrame, pl.LazyFrame] = None,
        fold: int = None,
        fold_col: str = "cv",
        feature_col: str = None,
        target_col: str = None,
        time_series_validation: bool = False,
    ):
        self.df = df
        self.df_val = df_val
        self.fold = fold
        self.fold_col = fold_col
        self.feature_col = feature_col
        self.target_col = target_col
        self.time_series_validation = time_series_validation

        # Split into train and test sets
        result = time_series_validation_filter(
            df=self.df,
            df_val=self.df_val,
            fold=self.fold,
            fold_col=self.fold_col,
            feature_col=self.feature_col,
            target_col=self.target_col,
            time_series_validation=self.time_series_validation,
        )
        (self.X_train, self.y_train, self.X_test, self.y_test) = result

        # Type of target variable:
        self.is_binary = get_column_dtype(self.y_train) in ["categorical", "binary"]

        # Either logistic or linear regression
        self.model = (
            fit_sm_logistic_regression(self.X_train, self.y_train)
            if self.is_binary
            else fit_sm_linear_regression(self.X_train, self.y_train)
        )
        self.sk_model = (
            fit_sk_logistic_regression(self.X_train, self.y_train)
            if self.is_binary
            else fit_sk_linear_regression(self.X_train, self.y_train)
        )

        # Test that the results are close:
        if np.round(self.model.params.iloc[0], 0) != np.round(
            self.sk_model.coef_[0][0], 0
        ):
            raise ValueError(
                f"The coefficient estimates from statsmodels {np.round(self.model.params.iloc[0], 2)} and sklearn {np.round(self.sk_model.coef_[0][0], 2)} are not close. This is likely due to a difference in the optimization algorithm used by each package. Try increasing the number of iterations in the statsmodels model."
            )

        # Pull stats from the fitted model object
        self.yhat_train = self.model.predict(self.X_train)
        self.yhat_test = self.model.predict(self.X_test)

        self.coef = self.model.params.iloc[0]
        self.pvalues = self.model.pvalues[0]
        self.aic = self.model.aic
        self.se = self.model.bse[0]
        # self.lower_ci = self.model.conf_int()[0][0]
        # self.upper_ci = self.model.conf_int()[0][1]
        self.n = self.model.nobs
        self.k = self.model.params.shape[0]

        self.sk_coef = self.sk_model.coef_

        # if get_column_dtype(self.y_train) in ["categorical", "binary"]:
        #     self.auc = roc_curve(self.y_test, self.yhat_test)
        #     self.prc = precision_recall_curve(self.y_test, self.yhat_test)
