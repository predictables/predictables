# from __future__ import annotations

# import numpy as np
# import pandas as pd
# import polars as pl
# from sklearn import metrics  # type: ignore
# from sklearn.preprocessing import StandardScaler  # type: ignore

# from predictables.univariate.src import (
#     fit_sk_linear_regression,
#     fit_sk_logistic_regression,
#     fit_sm_linear_regression,
#     fit_sm_logistic_regression,
#     remove_missing_rows,
# )
# from predictables.util import (
#     DebugLogger,
#     filter_df_by_cv_fold,
#     get_column_dtype,
#     to_pd_df,
#     to_pd_s,
#     to_pl_lf,
#     to_pl_s,
# )

# dbg = DebugLogger(working_file="_BaseModel.py")


# class Model:
#     """
#     A class to fit a model to a dataset. Used in the univariate analysis
#     to fit a simple model to each variable.
#     """

#     # Instance/type class attributes
#     df: pl.LazyFrame
#     df_val: pl.LazyFrame
#     fold_n: Optional[int]
#     fold_col: str
#     feature_col: str
#     target_col: str
#     time_series_validation: bool

#     X_train: pl.LazyFrame
#     y_train: pd.Series
#     X_test: pl.LazyFrame
#     y_test: pd.Series

#     is_binary: bool

#     model: Any
#     sk_model: Any

#     yhat_train: pd.Series
#     yhat_test: pd.Series

#     def __init__(
#         self,
#         df: Union[pd.DataFrame, pl.DataFrame, pl.LazyFrame],
#         df_val: Optional[Union[pd.DataFrame, pl.DataFrame, pl.LazyFrame]] = None,
#         fold_n: Optional[int] = None,
#         fold_col: str = "fold",
#         feature_col: str | None = None,
#         target_col: str | None = None,
#         time_series_validation: bool = False,
#     ) -> None:
#         dbg.msg(f"__init__ called on {self.__class__.__name__} with df and df_val.")
#         dbg.msg(f"feature_col: {feature_col}, target_col: {target_col}")
#         self.df = to_pl_lf(df)
#         self.df_val = to_pl_lf(df_val) if df_val is not None else to_pl_lf(df)
#         self.fold_n = fold_n
#         self.fold_col = fold_col
#         self.feature_col = (
#             feature_col if feature_col is not None else self.df.columns[1]
#         )
#         self.target_col = target_col if target_col is not None else self.df.columns[0]
#         self.time_series_validation = time_series_validation

#         # Remove rows with missing values
#         self.df = remove_missing_rows(self.df, self.feature_col, self.target_col)
#         self.df_val = remove_missing_rows(
#             self.df_val, self.feature_col, self.target_col
#         )

#         # Initialize results lazyframe
#         self.results = pl.DataFrame(
#             {
#                 "model": [self.__str__()],
#                 "model_name": [self.__repr__()],
#                 "fold": [
#                     (f"fold-{self.fold_n}" if self.fold_n is not None else "none")
#                 ],
#                 "feature": [self.feature_col],
#                 "feature_dtype": [
#                     get_column_dtype(
#                         self.df.select(self.feature_col).collect()[self.feature_col]
#                     )
#                 ],
#                 "target": [self.target_col],
#                 "target_dtype": [
#                     get_column_dtype(
#                         self.df.select(self.target_col).collect()[self.target_col]
#                     )
#                 ],
#             }
#         ).lazy()

#         # Split into train and test sets
#         if self.fold_n is None:
#             train = self.df
#             test = self.df_val

#         else:
#             train = filter_df_by_cv_fold(
#                 self.df,
#                 self.fold_n,
#                 self.df.select([self.fold_col]).collect().to_series(),
#                 self.time_series_validation,
#                 "train",
#                 "pl",
#             )
#             test = filter_df_by_cv_fold(
#                 self.df,
#                 self.fold_n,
#                 self.df.select([self.fold_col]).collect().to_series(),
#                 self.time_series_validation,
#                 "test",
#                 "pl",
#             )
#         self.X_train = train.select([self.feature_col])
#         self.y_train = train.select([self.target_col])
#         self.X_test = test.select([self.feature_col])
#         self.y_test = test.select([self.target_col])

#         self.scaler: Optional[StandardScaler] = None

#     def __post_init__(self) -> None:
#         """
#         Initializes the model and fits it to the input data.
#         """
#         # Type of target variable:
#         self.is_binary = get_column_dtype(self.GetY("train", dropna=True)) in [
#             "categorical",
#             "binary",
#         ]

#         # Either logistic or linear regression
#         if self.is_binary:
#             try:
#                 self.model = fit_sm_logistic_regression(
#                     self.GetX("train", dropna=True), self.GetY("train", dropna=True)
#                 )
#                 self.sk_model = fit_sk_logistic_regression(
#                     self.GetX("train", dropna=True), self.GetY("train", dropna=True)
#                 )
#             except np.linalg.LinAlgError:
#                 dbg.msg(
#                     f"LinAlgError caught for {self.feature_col}. "
#                     "Fitting model without missing rows, and removing 0s."
#                 )
#                 missing_zero_idx = (
#                     to_pd_df(self.GetX("train"))
#                     .iloc[:, 0]
#                     .fillna(0)
#                     .eq(0)
#                     .reset_index(drop=True)
#                 )
#                 X = (
#                     to_pd_df(self.GetX("train"))
#                     .reset_index(drop=True)
#                     .loc[~missing_zero_idx]
#                 )
#                 y = self.GetY("train").reset_index(drop=True)[~missing_zero_idx]

#                 try:
#                     self.model = fit_sm_logistic_regression(X, y)
#                     self.sk_model = fit_sk_logistic_regression(X, y)
#                 except Exception as e:
#                     dbg.msg(f"Error: {e}")  # type: ignore
#         else:
#             try:
#                 self.model = fit_sm_linear_regression(
#                     self.GetX("train", dropna=True), self.GetY("train", dropna=True)
#                 )
#                 self.sk_model = fit_sk_linear_regression(
#                     self.GetX("train", dropna=True), self.GetY("train", dropna=True)
#                 )
#             except np.linalg.LinAlgError:
#                 missing_zero_idx = (
#                     to_pd_df(self.GetX("train"))
#                     .iloc[:, 0]
#                     .fillna(0)
#                     .eq(0)
#                     .reset_index(drop=True)
#                 )
#                 X = (
#                     to_pd_df(self.GetX("train"))
#                     .reset_index(drop=True)
#                     .loc[~missing_zero_idx]
#                 )
#                 y = to_pd_s(self.GetY("train")).reset_index(drop=True)[
#                     ~missing_zero_idx
#                 ]
#                 self.model = fit_sm_linear_regression(X, y)
#                 self.sk_model = fit_sk_linear_regression(X, y)

#         self.yhat_train: Union[pd.Series[Any], pd.DataFrame[Any]] = self.predict(
#             self.GetX("train", dropna=True)
#         )  # type: ignore
#         self.yhat_test: Union[pd.Series[Any], pd.DataFrame[Any]] = self.predict(
#             self.GetX("test", dropna=True)
#         )  # type: ignore

#         # Pull stats from the fitted model object
#         self.results = self.results.with_columns(
#             pl.lit(self.model.params.iloc[0]).alias("coef")
#         )
#         self.results = self.results.with_columns(
#             pl.lit(self.model.pvalues.iloc[0]).alias("pvalues")
#         )
#         self.results = self.results.with_columns(pl.lit(self.model.aic).alias("aic"))
#         self.results = self.results.with_columns(
#             pl.lit(self.model.bse.iloc[0]).alias("se")
#         )
#         self.results = self.results.with_columns(
#             pl.lit(self.model.conf_int()[0].to_numpy()[0]).alias("lower_ci")
#         )
#         self.results = self.results.with_columns(
#             pl.lit(self.model.conf_int()[1]).alias("upper_ci")
#         )
#         self.results = self.results.with_columns(pl.lit(self.model.nobs).alias("n"))
#         self.results = self.results.with_columns(
#             pl.lit(self.model.params.shape[0]).alias("k")
#         )
#         self.results = self.results.with_columns(
#             pl.lit(self.sk_model.coef_).alias("sk_coef")
#         )

#         if self.is_binary:
#             self.results = self.results.with_columns(
#                 [
#                     pl.lit(
#                         metrics.accuracy_score(
#                             self.GetY("train"), self.GetYhat("train").round(0)
#                         )
#                     ).alias("acc_train"),
#                     pl.lit(
#                         metrics.accuracy_score(
#                             self.GetY("test"), self.GetYhat("test").round(0)
#                         )
#                     ).alias("acc_test"),
#                     pl.lit(
#                         metrics.f1_score(
#                             self.GetY("train"), self.GetYhat("train").round(0)
#                         )
#                     ).alias("f1_train"),
#                     pl.lit(
#                         metrics.f1_score(
#                             self.GetY("test"), self.GetYhat("test").round(0)
#                         )
#                     ).alias("f1_test"),
#                     pl.lit(
#                         metrics.recall_score(
#                             self.GetY("train"), self.GetYhat("train").round(0)
#                         )
#                     ).alias("recall_train"),
#                     pl.lit(
#                         metrics.recall_score(
#                             self.GetY("test"), self.GetYhat("test").round(0)
#                         )
#                     ).alias("recall_test"),
#                     pl.lit(
#                         metrics.log_loss(
#                             self.GetY("train"), self.GetYhat("train").round(0)
#                         )
#                     ).alias("logloss_train"),
#                     pl.lit(
#                         metrics.log_loss(
#                             self.GetY("test"), self.GetYhat("test").round(0)
#                         )
#                     ).alias("logloss_test"),
#                     pl.lit(
#                         metrics.roc_auc_score(
#                             self.GetY("train"), self.GetYhat("train").round(0)
#                         )
#                     ).alias("auc_train"),
#                     pl.lit(
#                         metrics.roc_auc_score(
#                             self.GetY("test"), self.GetYhat("test").round(0)
#                         )
#                     ).alias("auc_test"),
#                     pl.lit(
#                         metrics.precision_score(
#                             self.GetY("train"),
#                             self.GetYhat("train").round(0),
#                             zero_division=0,
#                         )
#                     ).alias("precision_train"),
#                     pl.lit(
#                         metrics.precision_score(
#                             self.GetY("test"),
#                             self.GetYhat("test").round(0),
#                             zero_division=0,
#                         )
#                     ).alias("precision_test"),
#                     pl.lit(
#                         metrics.matthews_corrcoef(
#                             self.GetY("train").replace(0, -1),
#                             self.GetYhat("train").round(0).replace(0, -1),
#                         )
#                     ).alias("mcc_train"),
#                     pl.lit(
#                         metrics.matthews_corrcoef(
#                             self.GetY("test").replace(0, -1),
#                             self.GetYhat("test").round(0).replace(0, -1),
#                         )
#                     ).alias("mcc_test"),
#                     pl.lit(
#                         metrics.roc_curve(
#                             self.GetY("train"), self.GetYhat("train").round(0)
#                         )
#                     ).alias("roc_curve_train"),
#                     pl.lit(
#                         metrics.roc_curve(
#                             self.GetY("test"), self.GetYhat("test").round(0)
#                         )
#                     ).alias("roc_curve_test"),
#                     pl.lit(
#                         metrics.precision_recall_curve(
#                             self.GetY("train"), self.GetYhat("train").round(0)
#                         )
#                     ).alias("pr_curve_train"),
#                     pl.lit(
#                         metrics.precision_recall_curve(
#                             self.GetY("test"), self.GetYhat("test").round(0)
#                         )
#                     ).alias("pr_curve_test"),
#                 ]
#             )

#     def __repr__(self) -> str:
#         return (
#             f"<Model{'_[CV-' if self.fold_n is not None else ''}"
#             f"{f'{self.fold_n}]' if self.fold_n is not None else ''}"
#             f"({'df' if self.df is not None else ''}"
#             f"{', df-val' if self.df_val is not None else ''})>"
#         )

#     def __str__(self) -> str:
#         return (
#             f"Model(df-val={'loaded' if self.df_val is not None else 'none'}, "
#             f"cv={f'fold-{self.fold_n}' if self.fold_n is not None else 'none'})"
#         )

#     def get(self, attr: str) -> Any:
#         """
#         Returns the value of the attribute.

#         Parameters
#         ----------
#         attr : str
#             The attribute to return.

#         Returns
#         -------
#         Any
#             The value of the attribute.
#         """
#         return self.results.select(attr).collect().item(0, 0)

#     def _fit_standardization(
#         self, X: Union[pd.Series, pl.Series, pd.DataFrame, pl.DataFrame, pl.LazyFrame]
#     ) -> None:
#         """
#         Fits a StandardScaler to the input data.

#         Parameters
#         ----------
#         X : Union[pd.Series, pl.Series, pd.DataFrame, pl.DataFrame, pl.LazyFrame]
#             The data to standardize.
#         """
#         if isinstance(X, pl.Series):
#             X = to_pd_s(X)
#         elif isinstance(X, (pl.DataFrame, pl.LazyFrame)):
#             X = to_pd_df(X)

#         if isinstance(X, pd.Series):
#             X = X.to_numpy().reshape(-1, 1) if X.shape[1] == 1 else X

#         # fit normalized data
#         self.scaler = StandardScaler()
#         self.scaler.fit(X.to_numpy())

#     def standardize(
#         self, X: Union[pd.Series, pl.Series, pd.DataFrame, pl.DataFrame, pl.LazyFrame]
#     ) -> Union[pd.Series, pd.DataFrame]:
#         """
#         Standardizes the input data.

#         Parameters
#         ----------
#         X : Union[pd.Series, pl.Series, pd.DataFrame, pl.DataFrame, pl.LazyFrame]
#             The data to standardize.

#         Returns
#         -------
#         pd.Series
#             The standardize data.
#         """
#         if not isinstance(
#             X, (pd.Series, pd.DataFrame, pl.Series, pl.DataFrame, pl.LazyFrame)
#         ):
#             raise ValueError(
#                 f"X must be a pandas or polars Series or DataFrame, not {type(X)}"
#             )

#         # Convert to pandas data types from polars if necessary
#         if isinstance(X, pl.Series):
#             X = to_pd_s(X)

#         if isinstance(X, (pl.DataFrame, pl.LazyFrame, pd.DataFrame)):
#             X = to_pd_df(X)

#         if self.scaler is None:
#             self._fit_standardization(X)

#         # Convert to data frame
#         if isinstance(X, pd.Series):
#             X = pd.DataFrame(X)

#         # Standardize and return the input data
#         if isinstance(self.scaler, StandardScaler):
#             return pd.DataFrame(
#                 self.scaler.transform(
#                     np.array(X.to_numpy()).reshape(-1, 1) if X.shape[1] == 1 else X
#                 ),  # type: ignore
#                 index=X.index,
#                 columns=X.columns,
#             )
#         else:
#             raise ValueError("Scaler is not fitted.")

#     def predict(
#         self,
#         X: Union[pd.Series, pl.Series, pd.DataFrame, pl.DataFrame, pl.LazyFrame],
#         name: str | None = None,
#     ) -> Union[pd.Series, pd.DataFrame]:
#         """
#         Predicts the target variable using the input data. If the univariate analysis
#         found a better normalization method than the unadjusted data, the input data
#         will be normalized before prediction.

#         Parameters
#         ----------
#         X : Union[pd.Series, pl.Series, pd.DataFrame, pl.DataFrame, pl.LazyFrame]
#             The data to predict.
#         name : str | None, optional
#             The name of the column to use for the predicted target variable. If None,
#             the name of the target variable from the univariate analysis with "_hat"
#             will be used.

#         Returns
#         -------
#         pd.Series
#             The predicted target variable.
#         """
#         # Normalize the input data
#         X = self.standardize(X)

#         # Convert X to pandas DataFrame if necessary
#         if not isinstance(X, (pd.DataFrame, pl.DataFrame, pl.LazyFrame)):
#             if isinstance(X, pl.Series):
#                 X = pd.DataFrame(to_pd_s(X))
#             elif isinstance(X, pd.Series):
#                 X = pd.DataFrame(X)
#             else:
#                 raise ValueError(
#                     f"X must be a pandas or polars Series or DataFrame, not {type(X)}"
#                 )

#         # Predict the target variable and return the result as a pandas Series
#         return pd.Series(
#             self.model.predict(X),
#             name=self.target_col + "_hat" if name is None else name,
#         )

#     def GetX(
#         self, train_test: str = "train", return_type: str = "pd", dropna: bool = False
#     ) -> Union[pd.DataFrame, pl.DataFrame]:
#         """
#         Returns the input data for the specified train or test set.

#         Parameters
#         ----------
#         train_test : str
#             The train or test set to return the input data for.
#         return_type : str
#             The type of data to return. Choices are "pd" for pandas or "pl" for polars.

#         Returns
#         -------
#         Union[pd.DataFrame, pl.DataFrame]
#             The input data for the specified train or test set.
#         """
#         if train_test == "train":
#             df = (
#                 to_pd_df(self.X_train)
#                 if return_type == "pd"
#                 else to_pl_lf(self.X_train)
#             )
#         elif train_test == "test":
#             df = to_pd_df(self.X_test) if return_type == "pd" else to_pl_lf(self.X_test)
#         else:
#             raise ValueError(f"train_test must be 'train' or 'test'. Got {train_test}.")

#         # replace -999 with NaN if necessary
#         if return_type == "pd":
#             df = df.replace(-999, np.nan)
#         else:
#             df = df.replace(-999, pl.NA)

#         # drop missing values if necessary
#         if dropna:
#             df = df.dropna() if return_type == "pd" else df.drop_nulls()

#         return df

#     def GetY(
#         self, train_test: str = "train", return_type: str = "pd", dropna: bool = False
#     ) -> pd.Series:
#         """
#         Returns the target variable for the specified train or test set.

#         Parameters
#         ----------
#         train_test : str
#             The train or test set to return the target variable for.

#         Returns
#         -------
#         pd.Series
#             The target variable for the specified train or test set.
#         """
#         if train_test == "train":
#             df = (
#                 to_pd_s(self.y_train.collect().to_series())
#                 if return_type == "pd"
#                 else to_pl_s(self.y_train.collect().to_series())
#             )
#         elif train_test == "test":
#             df(
#                 to_pd_s(self.y_test.collect().to_series())
#                 if return_type == "pd"
#                 else to_pl_s(self.y_test.collect().to_series())
#             )
#         else:
#             raise ValueError(f"train_test must be 'train' or 'test'. Got {train_test}.")

#         # drop missing values if necessary
#         if return_type == "pd":
#             df = df.to_frame().loc[self.GetX(train_test, return_type, dropna).index]
#         else:
#             X = (
#                 self.GetX(train_test, return_type)
#                 .with_context([pl.Series(df).alias("y")])
#                 .with_columns(
#                     [
#                         pl.col(self.GetX(train_test, return_type).columns[0])
#                         .replace(-999, pl.NA)
#                         .name.keep()
#                     ]
#                 )
#                 .filter(
#                     pl.col(self.GetX(train_test, return_type).columns[0]).is_not_nan()
#                 )
#                 .select([pl.col("y").name.keep()])
#             )
#             return X

#     def GetYhat(
#         self, train_test: str = "train", return_type: str = "pd"
#     ) -> Union[pd.Series, pd.DataFrame]:
#         """
#         Returns the predicted target variable for the specified train or test set.

#         Parameters
#         ----------
#         train_test : str
#             The train or test set to return the predicted target variable for.
#         return_type : str
#             The type of data to return. Choices are "pd" for pandas or "pl" for polars.

#         Returns
#         -------
#         Union[pd.Series, pd.DataFrame]
#             The predicted target variable for the specified train or test set.
#         """
#         if train_test == "train":
#             return self.yhat_train if return_type == "pd" else to_pl_s(self.yhat_train)
#         elif train_test == "test":
#             return self.yhat_test if return_type == "pd" else to_pl_s(self.yhat_test)
#         else:
#             raise ValueError(f"train_test must be 'train' or 'test'. Got {train_test}.")
