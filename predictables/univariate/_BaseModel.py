from typing import Any, Optional, Union

import numpy as np
import pandas as pd
import polars as pl
import sklearn.metrics as metrics  # type: ignore
from sklearn.preprocessing import StandardScaler  # type: ignore

from predictables.util import get_column_dtype, to_pd_df, to_pd_s

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

    X_train: pd.DataFrame
    y_train: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series

    is_binary: bool

    model: Any
    sk_model: Any

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
        (self.X_train, self.y_train, self.X_test, self.y_test) = (
            time_series_validation_filter(
                df=self.df,
                df_val=self.df_val,
                fold=self.fold_n,
                fold_col=self.fold_col,
                feature_col=self.feature_col,
                target_col=self.target_col,
                time_series_validation=self.time_series_validation,
            )
        )

        self.scaler: Optional[StandardScaler] = None

        # Type of target variable:
        self.is_binary = get_column_dtype(self.y_train) in ["categorical", "binary"]

        # Either logistic or linear regression
        if self.is_binary:
            self.model = fit_sm_logistic_regression(self.X_train, self.y_train)
            self.sk_model = fit_sk_logistic_regression(self.X_train, self.y_train)
        else:
            self.model = fit_sm_linear_regression(self.X_train, self.y_train)
            self.sk_model = fit_sk_linear_regression(self.X_train, self.y_train)

        self.yhat_train: Union[pd.Series[Any], pd.DataFrame[Any]] = self.predict(self.X_train)  # type: ignore
        self.yhat_test: Union[pd.Series[Any], pd.DataFrame[Any]] = self.predict(self.X_test)  # type: ignore

        # Pull stats from the fitted model object
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
            self.acc_train = metrics.accuracy_score(
                self.y_train, self.yhat_train.round(0)
            )
            self.acc_test = metrics.accuracy_score(self.y_test, self.yhat_test.round(0))
            self.f1_train = metrics.f1_score(self.y_train, self.yhat_train.round(0))
            self.f1_test = metrics.f1_score(self.y_test, self.yhat_test.round(0))
            self.recall_train = metrics.recall_score(
                self.y_train, self.yhat_train.round(0)
            )
            self.recall_test = metrics.recall_score(
                self.y_test, self.yhat_test.round(0)
            )

            self.logloss_train = metrics.log_loss(
                self.y_train, self.yhat_train.round(0)
            )
            self.logloss_test = metrics.log_loss(self.y_test, self.yhat_test.round(0))
            self.auc_train = metrics.roc_auc_score(
                self.y_train, self.yhat_train.round(0)
            )
            self.auc_test = metrics.roc_auc_score(self.y_test, self.yhat_test.round(0))

            self.precision_train = metrics.precision_score(
                self.y_train, self.yhat_train.round(0), zero_division=0
            )
            self.precision_test = metrics.precision_score(
                self.y_test, self.yhat_test.round(0), zero_division=0
            )
            self.mcc_train = metrics.matthews_corrcoef(
                self.y_train.replace(0, -1), self.yhat_train.round(0).replace(0, -1)
            )
            self.mcc_test = metrics.matthews_corrcoef(
                self.y_test.replace(0, -1), self.yhat_test.round(0).replace(0, -1)
            )

            self.roc_curve_train = metrics.roc_curve(
                self.y_train, self.yhat_train.round(0)
            )
            self.roc_curve_test = metrics.roc_curve(
                self.y_test, self.yhat_test.round(0)
            )
            self.pr_curve_train = metrics.precision_recall_curve(
                self.y_train, self.yhat_train.round(0)
            )
            self.pr_curve_test = metrics.precision_recall_curve(
                self.y_test, self.yhat_test.round(0)
            )

    def __repr__(self) -> str:
        return f"<Model{'_[CV-' if self.fold_n is not None else ''}{f'{self.fold_n}]' if self.fold_n is not None else ''}({'df' if self.df is not None else ''}{', df-val' if self.df_val is not None else ''})>"

    def __str__(self) -> str:
        return f"Model(df-val={'loaded' if self.df_val is not None else 'none'}, cv={f'fold-{self.fold_n}' if self.fold_n is not None else 'none'})"

    def _fit_standardization(
        self,
        X: Union[pd.Series, pl.Series, pd.DataFrame, pl.DataFrame, pl.LazyFrame],
    ) -> None:
        """
        Fits a StandardScaler to the input data.

        Parameters
        ----------
        X : Union[pd.Series, pl.Series, pd.DataFrame, pl.DataFrame, pl.LazyFrame]
            The data to standardize.
        """
        if isinstance(X, pl.Series):
            X = to_pd_s(X)
        elif isinstance(X, (pl.DataFrame, pl.LazyFrame)):
            X = to_pd_df(X)

        if isinstance(X, pd.Series):
            X = X.values.reshape(-1, 1) if X.shape[1] == 1 else X  # type: ignore

        # fit normalized data
        self.scaler = StandardScaler()
        self.scaler.fit(X)

    def standardize(
        self, X: Union[pd.Series, pl.Series, pd.DataFrame, pl.DataFrame, pl.LazyFrame]
    ) -> Union[pd.Series, pd.DataFrame]:
        """
        Standardizes the input data.

        Parameters
        ----------
        X : Union[pd.Series, pl.Series, pd.DataFrame, pl.DataFrame, pl.LazyFrame]
            The data to standardize.

        Returns
        -------
        pd.Series
            The standardize data.
        """
        if not isinstance(
            X, (pd.Series, pd.DataFrame, pl.Series, pl.DataFrame, pl.LazyFrame)
        ):
            raise ValueError(
                f"X must be a pandas or polars Series or DataFrame, not {type(X)}"
            )

        # Convert to pandas data types from polars if necessary
        if isinstance(X, pl.Series):
            X = to_pd_s(X)

        if isinstance(X, (pl.DataFrame, pl.LazyFrame, pd.DataFrame)):
            X = to_pd_df(X)

        if self.scaler is None:
            self._fit_standardization(X)

        # Convert to data frame
        if isinstance(X, pd.Series):
            X = pd.DataFrame(X)

        # Standardize and return the input data
        return pd.DataFrame(
            self.scaler.transform(np.array(X.values).reshape(-1, 1) if X.shape[1] == 1 else X),  # type: ignore
            index=X.index,
            columns=X.columns,
        )

    def predict(
        self,
        X: Union[pd.Series, pl.Series, pd.DataFrame, pl.DataFrame, pl.LazyFrame],
        name: Optional[str] = None,
    ) -> Union[pd.Series, pd.DataFrame]:
        """
        Predicts the target variable using the input data. If the univariate analysis
        found a better normalization method than the unadjusted data, the input data
        will be normalized before prediction.

        Parameters
        ----------
        X : Union[pd.Series, pl.Series, pd.DataFrame, pl.DataFrame, pl.LazyFrame]
            The data to predict.
        name : Optional[str], optional
            The name of the column to use for the predicted target variable. If None,
            the name of the target variable from the univariate analysis with "_hat"
            will be used.

        Returns
        -------
        pd.Series
            The predicted target variable.
        """
        # Normalize the input data
        X = self.standardize(X)

        # Predict the target variable and return the result as a pandas Series
        return pd.Series(
            self.model.predict(X),
            index=X.index,
            name=self.target_col + "_hat" if name is None else name,
        )
