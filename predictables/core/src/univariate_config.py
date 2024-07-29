"""Configuration for univariate analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generator, List, Protocol, Tuple

import pandas as pd
import polars as pl


@dataclass
class UnivariateConfigInterface(Protocol):
    @property
    def df(self) -> pl.LazyFrame:
        """Return the training lazyframe."""
        ...

    @property
    def df_val(self) -> pl.LazyFrame:
        """Return the validation lazyframe."""
        ...

    @property
    def target(self) -> str:
        """Return the column name of the target variable."""
        ...

    @property
    def features(self) -> list[str]:
        """Return a list of the names of the features from the analysis."""
        ...

    @property
    def X_y_by_cv(self) -> Generator[tuple[pd.DataFrame, pd.DataFrame], None, None]:
        """Return a generator that yields (X_train, X_test, y_train, y_test) at each iteration."""
        ...


class UnivariateConfig:
    """Configure options for a univariate analysis."""

    def __init__(
        self,
        model_name: str | None = None,
        df_train: pl.LazyFrame | None = None,
        df_val_: pl.LazyFrame | None = None,
        target_column_name: str | None = None,
        feature_column_names: List[str] | None = None,
        time_series_validation: bool = True,
        cv_column_name: str = "cv",
        cv_folds: pl.Series | None = None
    ): 

        # Raise a ValueError if the dataframe is empty
        if df_train.collect().shape[0] == 0:
            raise ValueError(
                "Empty dataframes are not supported. Training data is empty."
            )

        self._model_name = model_name
        self._df_train = df_train
        self._df_val = df_val
        self._target_column_name = target_column_name
        self._feature_column_names = feature_column_names
        self._time_series_validation = time_series_validation
        self._cv_column_name = cv_column_name
        self._cv_folds = cv_folds



    @property
    def model_name(self) -> str:
        """Return the model name or a default if none was provided."""
        return self._model_name if self._model_name is not None else "univariate-analysis-model"

    @model_name.setter
    def model_name(self, name:str) -> None:
        self._model_name = name

    @property
    def df(self) -> pl.LazyFrame:
        """Return the training data, coerced to lazyframe if needed."""
        return (
            to_pl_lf(self._df_train)
            if not isinstance(self._df_train, pl.LazyFrame)
            else self._df_train
        )

    @df.setter
    def df(self, df_: pl.LazyFrame) -> None:
        """Set the training dataframe."""
        self._df_train = df_ if isinstance(df_, pl.LazyFrame) else to_pl_lf(df_)

    @property
    def df_val(self) -> pl.LazyFrame:
        """Return the validation dataset, if provided, and the training dataset if not."""
        if self._df_val is None:
            return self.df
        
        return (
            to_pl_lf(self._df_val)
            if not isinstance(self._df_val, pl.LazyFrame)
            else self._df_val
        )

    @df_val.setter
    def df_val(self, df_: pl.LazyFrame) -> None:
        """Set the validation dataframe."""
        self._df_val = df_ if isinstance(df_, pl.LazyFrame) else to_pl_lf(df_)

    @property
    def target(self) -> str:
        """Return the target variable name or a default."""
        return self._target_column_name if self._target_column_name is not None else "evolve_hit_count"

    @target.setter
    def target(self, name: str) -> None:
        """Set the target variable for the analysis."""
        self._target_column_name = name

    @property
    def features(self) -> list[str]:
        """Return a list of the features used in the analysis."""
        if isinstance(self._feature_column_names, list):
            return self._feature_column_names
        elif isinstance(self._feature_column_names, str):
            return [self._feature_column_names]
        elif self._feature_column_names is None:
            return [c for c in self.df.columns if c not in ['y', self.target, self.cv_column_name]]
        else:
            raise ValueError(f"Expected self._feature_column_names to be either a string or a list of strings, but got {type(self._feature_column_names)}")

    @features.setter
    def features(self, features_: list[str] | None) -> None:
        """Set the feature columns."""
        self._feature_column_names = features_

    def reset_features(self) -> None:
        """Reset the features property to pull them directly from the training data."""
        self.features(None)

    @property
    def time_series_validation(self) -> bool:
        """Return a boolean to indicate whether or not we are using time-series cross-validation or not."""
        return self._time_series_validation

    @time_series_validation.setter
    def time_series_validation(self, use_ts_validation: bool) -> None:
        """Set the time_series_validation property."""
        self._time_series_validation = use_ts_validation

    def toggle_time_series_validation(self) -> None:
        """Toggle the time-series validation indicator."""
        self.time_series_validation(~self.time_series_validation)

    @property
    def cv_column_name(self) -> str:
        """Return the column containing the cross-validation fold labels."""
        if self._cv_column_name not in self.features:
            raise ValueError(f"Expected CV column name {self._cv_column_name} is not in the features:\n{self.features}.")
        
        return self._cv_column_name

    @cv_column_name.setter
    def cv_column_name(self, name: str) -> None:
        """Set the cv column name."""
        if name not in self.features:
            raise ValueError(f"Expected CV column name {name} to be in the features:\n{self.features}.")

        self._cv_column_name = name

    @property
    def cv_folds(self) -> pd.Series:
        if self._cv_folds is not None:
            return to_pd_s(self._cv_folds)
        
        return (
            self.df
            .select(pl.col(self._cv_column_name))
            .collect()
            .to_series()
            .to_pandas()
        )

    @property
    def X_y_by_cv(self) -> Generator[tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series], None, None]:
        """Return the training and validation data as a generator."""
        for i in range(self.cv_folds.min() + 1, self.cv_folds.max() + 1):
            df_train, df_val = (
                self.filter_for_time_series(self.df, i)
                if self.time_series_validation
                else self.filter_for_non_time_series(self.df, i)
            )
            X_train = (
                df_train.select(self.features)
                .collect()
                .to_pandas()
            )
            y_train = (
                df_train.select(self.target)
                .collect()
                .to_numpy()
                .ravel()
            )
            X_test = (
                df_val.select(self.features)
                .collect()
                .to_pandas()
            )
            y_test = (
                df_val.select(self.target)
                .collect()
                .to_numpy()
                .ravel()
            )
            yield X_train, X_test, y_train, y_test

    def filter_for_time_series(
        self, X: pl.LazyFrame, fold: int
    ) -> Tuple[pl.LazyFrame, pl.LazyFrame]:
        """Filter the training and testing data for a particular fold, given time series validation."""
        return X.filter(pl.col(self.cv_column_name) < fold), X.filter(
            pl.col(self.cv_column_name) == fold
        )

    def filter_for_non_time_series(
        self, X: pl.LazyFrame, fold: int
    ) -> Tuple[pl.LazyFrame, pl.LazyFrame]:
        """Filter the training and testing data for a particular fold, given non-time series validation."""
        return X.filter(pl.col(self.cv_column_name) != fold), X.filter(
            pl.col(self.cv_column_name) == fold
        )
