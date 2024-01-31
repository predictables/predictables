from typing import Any, Optional, Union

import pandas as pd
import polars as pl
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from predictables.util import get_column_dtype, to_pd_df

from .src import (
    fit_sk_linear_regression,
    fit_sk_logistic_regression,
    fit_sm_linear_regression,
    fit_sm_logistic_regression,
    time_series_validation_filter,
)


class Model:
    """
    A class to fit a model to a dataset. Used in the univariate analysis
    to fit a simple model to each variable.
    """

    # Instance/type class attributes
    df: pd.DataFrame
    df_val: pd.DataFrame
    fold_n: Optional[int]
    fold_col: str
    feature_col: str
    target_col: str
    time_series_validation: bool

    X_train: pd.Series
    y_train: pd.Series
    X_test: pd.Series
    y_test: pd.Series

    is_binary: bool

    model: Any  # statsmodels model object -- either GLM for logistic or OLS for linear
    sk_model: (
        Any  # sklearn model object -- either LogisticRegression or LinearRegression
    )

    yhat_train: pd.Series
    yhat_test: pd.Series

    coef: float
    pvalues: float
    aic: float
    se: float
    lower_ci: float
    upper_ci: float
    n: int
    k: int
    sk_coef: float

    # optional because only classification models have these
    acc_train: Optional[float]
    acc_test: Optional[float]
    f1_train: Optional[float]
    f1_test: Optional[float]
    recall_train: Optional[float]
    recall_test: Optional[float]
    logloss_train: Optional[float]
    logloss_test: Optional[float]
    auc_train: Optional[float]
    auc_test: Optional[float]
    precision_train: Optional[float]
    precision_test: Optional[float]
    mcc_train: Optional[float]
    mcc_test: Optional[float]
    roc_curve_train: Optional[float]
    roc_curve_test: Optional[float]
    pr_curve_train: Optional[float]
    pr_curve_test: Optional[float]

    def __init__(
        self,
        df: Union[pd.DataFrame, pl.DataFrame, pl.LazyFrame],
        df_val: Optional[Union[pd.DataFrame, pl.DataFrame, pl.LazyFrame]] = None,
        fold_n: Optional[int] = None,
        fold_col: str = "cv",
        feature_col: Optional[str] = None,
        target_col: Optional[str] = None,
        time_series_validation: bool = False,
    ) -> None:
        self.df = to_pd_df(df)
        self.df_val = to_pd_df(df_val) if df_val is not None else to_pd_df(df)
        self.fold_n = fold_n
        self.fold_col = fold_col
        self.feature_col = (
            feature_col if feature_col is not None else self.df.columns[1]
        )
        self.target_col = target_col if target_col is not None else self.df.columns[0]
        self.time_series_validation = time_series_validation

        # Split into train and test sets
        result: tuple = time_series_validation_filter(
            df=self.df,
            df_val=self.df_val,
            fold=self.fold_n,
            fold_col=self.fold_col,
            feature_col=self.feature_col,
            target_col=self.target_col,
            time_series_validation=self.time_series_validation,
        )
        (self.X_train, self.y_train, self.X_test, self.y_test) = result

        # Type of target variable:
        self.is_binary = get_column_dtype(self.y_train) in ["categorical", "binary"]

        # Either logistic or linear regression
        X_arr: Any = self.X_train.to_numpy().reshape(-1, 1)
        if self.is_binary:
            self.model = fit_sm_logistic_regression(X_arr, self.y_train)
            self.sk_model = fit_sk_logistic_regression(X_arr, self.y_train)
        else:
            self.model = fit_sm_linear_regression(X_arr, self.y_train)
            self.sk_model = fit_sk_linear_regression(X_arr, self.y_train)

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
            self.acc_train = accuracy_score(self.y_train, self.yhat_train.round(0))
            self.acc_test = accuracy_score(self.y_test, self.yhat_test.round(0))
            self.f1_train = f1_score(self.y_train, self.yhat_train.round(0))
            self.f1_test = f1_score(self.y_test, self.yhat_test.round(0))
            self.recall_train = recall_score(self.y_train, self.yhat_train.round(0))
            self.recall_test = recall_score(self.y_test, self.yhat_test.round(0))

            self.logloss_train = log_loss(self.y_train, self.yhat_train.round(0))
            self.logloss_test = log_loss(self.y_test, self.yhat_test.round(0))
            self.auc_train = roc_auc_score(self.y_train, self.yhat_train.round(0))
            self.auc_test = roc_auc_score(self.y_test, self.yhat_test.round(0))

            self.precision_train = precision_score(
                self.y_train, self.yhat_train.round(0), zero_division=0
            )
            self.precision_test = precision_score(
                self.y_test, self.yhat_test.round(0), zero_division=0
            )
            self.mcc_train = matthews_corrcoef(
                self.y_train.replace(0, -1), self.yhat_train.round(0).replace(0, -1)
            )
            self.mcc_test = matthews_corrcoef(
                self.y_test.replace(0, -1), self.yhat_test.round(0).replace(0, -1)
            )

            self.roc_curve_train = roc_curve(self.y_train, self.yhat_train.round(0))
            self.roc_curve_test = roc_curve(self.y_test, self.yhat_test.round(0))
            self.pr_curve_train = precision_recall_curve(
                self.y_train, self.yhat_train.round(0)
            )
            self.pr_curve_test = precision_recall_curve(
                self.y_test, self.yhat_test.round(0)
            )

    def __repr__(self) -> str:
        return f"<Model{'_[CV-' if self.fold_n is not None else ''}{f'{self.fold_n}]' if self.fold_n is not None else ''}({'df' if self.df is not None else ''}{', df-val' if self.df_val is not None else ''})>"

    def __str__(self) -> str:
        return f"Model(df-val={'loaded' if self.df_val is not None else 'none'}, cv={f'fold-{self.fold_n}' if self.fold_n is not None else 'none'})"
