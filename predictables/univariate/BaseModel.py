from typing import Union

import pandas as pd
import polars as pl
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    matthews_corrcoef,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_curve,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from predictables.util import get_column_dtype

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

        # Pull stats from the fitted model object
        self.yhat_train = self.model.predict(self.X_train)
        self.yhat_test = self.model.predict(self.X_test)

        self.coef = self.model.params.iloc[0]
        self.pvalues = self.model.pvalues.iloc[0]
        self.aic = self.model.aic
        self.se = self.model.bse.iloc[0]
        self.lower_ci = self.model.conf_int()[0].values[0]
        self.upper_ci = self.model.conf_int()[1].values[0]
        self.n = self.model.nobs
        self.k = self.model.params.shape[0]

        self.sk_coef = self.sk_model.coef_

        if self.is_binary:
            setattr(
                self,
                "acc_train",
                accuracy_score(self.y_train, self.yhat_train.round(0)),
            )
            setattr(
                self,
                "acc_test",
                accuracy_score(self.y_test, self.yhat_test.round(0)),
            )
            setattr(self, "f1_train", f1_score(self.y_train, self.yhat_train.round(0)))
            setattr(self, "f1_test", f1_score(self.y_test, self.yhat_test.round(0)))
            setattr(
                self,
                "recall_train",
                recall_score(self.y_train, self.yhat_train.round(0)),
            )
            setattr(
                self,
                "recall_test",
                recall_score(self.y_test, self.yhat_test.round(0)),
            )

            setattr(
                self, "logloss_train", log_loss(self.y_train, self.yhat_train.round(0))
            )
            setattr(
                self, "logloss_test", log_loss(self.y_test, self.yhat_test.round(0))
            )
            setattr(
                self, "auc_train", roc_auc_score(self.y_train, self.yhat_train.round(0))
            )
            setattr(
                self, "auc_test", roc_auc_score(self.y_test, self.yhat_test.round(0))
            )

            setattr(
                self,
                "precision_train",
                precision_score(self.y_train, self.yhat_train.round(0)),
            )
            setattr(
                self,
                "precision_test",
                precision_score(self.y_test, self.yhat_test.round(0)),
            )
            setattr(
                self,
                "mcc_train",
                matthews_corrcoef(
                    self.y_train.replace(0, -1), self.yhat_train.round(0).replace(0, -1)
                ),
            )
            setattr(
                self,
                "mcc_test",
                matthews_corrcoef(
                    self.y_test.replace(0, -1), self.yhat_test.round(0).replace(0, -1)
                ),
            )

            setattr(
                self,
                "roc_curve_train",
                roc_curve(self.y_train, self.yhat_train.round(0)),
            )
            setattr(
                self, "roc_curve_test", roc_curve(self.y_test, self.yhat_test.round(0))
            )
            setattr(
                self,
                "pr_curve_train",
                precision_recall_curve(self.y_train, self.yhat_train.round(0)),
            )
            setattr(
                self,
                "pr_curve_test",
                precision_recall_curve(self.y_test, self.yhat_test.round(0)),
            )

        # else:
        #     setattr(self, "mse", mean_squared_error(self.y_test, self.yhat_test))
        #     setattr(self, "mae", mean_absolute_error(self.y_test, self.yhat_test))
        #     setattr(self, "r2", r2_score(self.y_test, self.yhat_test))

        # # if get_column_dtype(self.y_train) in ["categorical", "binary"]:
        # #     self.auc = roc_curve(self.y_test, self.yhat_test)
        # #     self.prc = precision_recall_curve(self.y_test, self.yhat_test)

    def __repr__(self):
        return f"<Model{'_[CV-' if self.fold is not None else ''}{f'{self.fold}]' if self.fold is not None else ''}({'df' if self.df is not None else ''}{', df-val' if self.df_val is not None else ''})>"

    def __str__(self):
        return f"Model(df-val={'loaded' if self.df_val is not None else 'none'}, cv={f'fold-{self.fold}' if self.fold is not None else 'none'})"
