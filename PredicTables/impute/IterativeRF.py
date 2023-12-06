"""
This module implements an interative random forest imputation algorithm:

1. Start with the current dataset, X and impute numeric columns with median and categorical columns with mode to 
    create X_imputed.
2. Repeat until convergence: 
    2a. Fit a random forest regressor to each numeric column in `X_imputed` with missing values,
        using the other columns as predictors.
    2b. Fit a random forest classifier to each categorical column with missing values, using the
        other columns as predictors.
    2c. For each column with missing values, predict the missing values using the fitted random
        forest model.
    2d. Update the original imputation with the (predicted imputed values) * (learning rate).
    2e. Repeat until convergence or until the maximum number of iterations is reached.

.. module:: IterativeRF
:platform: Unix, Windows
:synopsis: Iterative random forest missing value imputation algorithm.
:author: Andy Weaver
:email: andrew_weaver@cinfin.com
:date: 2023-11-09
:version: 0.1

.. moduleauthor:: Andy Weaver <andrew_weaver@cinfin.com>

.. rubric:: Contents

- :ref:`section1`
- :ref:`section2`
- :ref:`section3`

.. rubric:: Usage

.. code-block:: python

# Example usage code here

.. rubric:: Sections

.. \_section1:

## Section 1

Section 1 description.

.. \_section2:

## Section 2

Section 2 description.

.. \_section3:

## Section 3

Section 3 description.
"""

from numpy import ndarray
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from skl2onnx import to_onnx

import numpy as np
import pandas as pd
import polars as pl

from tqdm import tqdm

import json
from testing import ImputeTester


from typing import Union

from .src.get_cv_folds import get_cv_folds
from .src.get_missing_data_mask import get_missing_data_mask
from .src.initial_impute import initial_impute
from .src.get_rf_hyperparameters import set_rf_params

# Helper type for 2D input data, which can be a pandas DataFrame, polars DataFrame, or a numpy array.
DataFrameType = Union[pd.DataFrame, pl.DataFrame, np.ndarray]

# Helper type for more general array-like input data, which can be a pandas DataFrame, polars DataFrame,
# polars Series, pandas Series, or a numpy array.
ArrayLikeType = Union[
    pd.DataFrame, pl.DataFrame, pl.LazyFrame, pl.Series, pd.Series, np.ndarray
]


class IterativeRF(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        X: DataFrameType = None,
        learning_rate: float = 0.1,
        n_iter: int = 10,
        tol: float = 1e-3,
        best_hyperparameters: dict = None,
        best_models: dict = None,
        random_state: int = 42,
        debug: bool = False,
        cv_eval: bool = True,
        cv_folds: int = 5,
        rf_hyperparameters: Union[
            str, dict, RandomForestClassifier, RandomForestRegressor
        ] = "default",
    ):
        """
        Initialize the IterativeRF class.
        """
        self.X = X
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.tol = tol
        self.random_state = random_state

        self.best_hyperparameters = (
            {} if best_hyperparameters is None else best_hyperparameters
        )
        self.best_models = {} if best_models is None else best_models
        self.missing_mask = None
        self.debug = debug
        self.cv_eval = cv_eval
        self.cv_folds = cv_folds
        self.tester = ImputeTester()

        self.bestX = initial_impute(self.X.copy())

        # If we are doing CV evaluation, assign each row of X to a fold
        if self.cv_eval:
            self.X_fold = pd.DataFrame(
                {"fold": get_cv_folds(self.X, n_folds=self.cv_folds)}
            )
            self.X_fold.index = self.X.index

        self.rf_hyperparameters = set_rf_params(rf_hyperparameters)

    def _missing_mask(self, X=None) -> pl.DataFrame:
        """Produces a dataframe with a boolean mask for missing values.

        :param[Union[pl.DataFrame, pd.DataFrame, pl.LazyFrame, pd.Series, pl.Series]] X: A dataframe. Will be coerced to a polars lazy frame.

        :return[pl.DataFrame]: A dataframe with a boolean mask for missing values.
        """
        # If X is None, use self.X
        if X is None:
            X = self.bestX

        # Get the missing mask
        missing_mask = get_missing_data_mask(X)

        return missing_mask

    def _set_hyperparameters(self, col=None, **kwargs):
        """
        Sets the hyperparameters for the RandomForest estimator.
        """
        # If a column isn't passed, set the hyperparameters for all columns
        if col is None:
            for col in self.X.columns.tolist():
                # if col is not in the best hyperparameters, add it
                if col not in self.best_hyperparameters.keys():
                    self.best_hyperparameters[col] = {}

                # Set the hyperparameters
                self.best_hyperparameters[col].update(kwargs)

                # Set (but don't fit) the best model
                if self.X[col].dtype in [np.float64, np.int64]:
                    self.best_models[col] = RandomForestRegressor(
                        **self.best_hyperparameters[col]
                    )
                else:
                    self.best_models[col] = RandomForestClassifier(
                        **self.best_hyperparameters[col]
                    )

        # Otherwise set the hyperparameters for the specified column
        else:
            # Get the column name and index
            if isinstance(col, str):
                col_name = col
                col_idx = self.bestX.columns.tolist().index(col_name)
            else:
                col_name = self.bestX.columns.tolist()[col]
                col_idx = col

            # if col is not in the best hyperparameters, add it
            if col not in self.best_hyperparameters.keys():
                self.best_hyperparameters[col] = {}

            # Set the hyperparameters
            self.best_hyperparameters[col].update(kwargs)

            # Set (but don't fit) the best model
            if self.X[col].dtype in [np.float64, np.int64]:
                self.best_models[col] = RandomForestRegressor(
                    **self.best_hyperparameters[col]
                )
            else:
                self.best_models[col] = RandomForestClassifier(
                    **self.best_hyperparameters[col]
                )

    def _find_best_hyperparameters_col(
        self,
        X=None,
        col=None,
        param_distributions=None,
        overwrite_best=False,
        numeric_scoring="neg_mean_squared_error",
        categorical_scoring="accuracy",
        cv=5,
        n_iter=50,
        n_jobs=-1,
        verbose=1,
        warm_start=True,
        random_state=None,
    ):
        """
        Finds and stores the best hyperparameters for the RandomForest estimator for a given column.
        """
        # If X is None, use self.X
        if X is None:
            X = self.bestX

        # Get the column name and index
        if col is not None:
            if isinstance(col, str):
                col_name = col
                col_idx = X.columns.tolist().index(col_name)
            else:
                col_name = X.columns.tolist()[col]
                col_idx = col

        # If the column already has tested for hyperparameters and overwrite is False, do nothing
        if not overwrite_best and col_name in self.best_hyperparameters:
            return

        # Get the column data type and set up the appropriate estimator and parameters
        col_dtype = X[col_name].dtype
        estimator, scoring, default_param = self._setup_estimator_and_scoring(
            col_dtype, numeric_scoring, categorical_scoring
        )

        # Ensure parameter distributions are provided
        if param_distributions is None:
            param_distributions = default_param

        # Prepare the features (X) and target (y) for fitting the model
        y = X[col_name]
        X_features = X.drop(columns=[col_name])

        # Impute missing values in features
        X_features_imputed = initial_impute(X_features)

        # Create the RandomizedSearchCV object
        clf = RandomizedSearchCV(
            estimator=estimator,
            param_distributions=param_distributions,
            scoring=scoring,
            cv=cv,
            n_iter=n_iter,
            verbose=verbose,
            random_state=random_state
            if random_state is not None
            else self.random_state,
            n_jobs=n_jobs,
        )

        # Fit the model
        clf.fit(X_features_imputed, y)

        # Store the best parameters
        self._set_hyperparameters(col_name, **clf.best_params_)

        if self.debug:
            print(f"Best hyperparameters for column {col_name}: {clf.best_params_}")

        # Store the best model
        self.best_models[col_name] = clf.best_estimator_

    def _setup_estimator_and_scoring(
        self,
        col_dtype,
        numeric_scoring,
        categorical_scoring,
        warm_start=True,
        random_state=None,
    ):
        """
        Sets up the appropriate estimator and scoring based on the column data type.
        """
        if col_dtype in [np.float64, np.int64]:
            estimator = RandomForestRegressor(
                random_state=random_state
                if random_state is not None
                else self.random_state,
                warm_start=warm_start,
                n_jobs=-1,
            )
            scoring = numeric_scoring
            default_param_distributions = regression_grid()

        else:
            estimator = RandomForestClassifier(
                random_state=random_state
                if random_state is not None
                else self.random_state,
                warm_start=warm_start,
                n_jobs=-1,
            )
            scoring = categorical_scoring
            default_param_distributions = classification_grid()

        return estimator, scoring, default_param_distributions

    def _find_best_hyperparameters(
        self,
        X=None,
        param_distributions=None,
        overwrite_best=False,
        numeric_scoring="neg_mean_squared_error",
        categorical_scoring="accuracy",
        cv=5,
        n_iter=50,
        n_jobs=-1,
        verbose=0,
        random_state=None,
    ):
        """
        Finds and stores the best hyperparameters for the RandomForest estimator for each column.
        """
        # If X is None, use self.X
        if X is None:
            X = self.bestX

        # Loop through each column and find the best hyperparameters
        for col in tqdm(range(X.shape[1]), desc="Finding best hyperparameters..."):
            self._find_best_hyperparameters_col(
                X=X,
                col=col,
                param_distributions=param_distributions,
                overwrite_best=overwrite_best,
                numeric_scoring=numeric_scoring,
                categorical_scoring=categorical_scoring,
                cv=cv,
                n_iter=n_iter,
                n_jobs=n_jobs,
                verbose=verbose,
                random_state=random_state,
            )

    def _fit_model(self, X=None, target_col=None):
        """
        Fits a RandomForest model to a single column. All columns except the target column
        are used as features.
        """
        # If X is None, use self.X
        if X is None:
            X = self.bestX

        # Get the column name and index
        col = target_col
        if col is not None:
            if isinstance(col, str):
                col_name = col
                col_idx = X.columns.tolist().index(col_name)
            else:
                col_name = X.columns.tolist()[col]
                col_idx = col

        # Get the column data
        y = X[col_name]
        X = X.drop(columns=[col_name])

        #

    def _predict_missing_values(self, X=None, col=None, overwrite_best=False):
        """
        Predicts missing values in a single column using the best model for that column.
        """
        # If X is None, use self.X
        if X is None:
            X = self.bestX

        # Get the column name
        if isinstance(col, str):
            col_name = col
            col_idx = X.columns.tolist().index(col_name)
        else:
            col_name = X.columns.tolist()[col]
            col_idx = col

        # If the column already has tested for hyperparameters, do nothing
        if (col_name in self.best_hyperparameters.keys()) & (overwrite_best is False):
            pass

        # Otherwise find the best hyperparameters using a RandomizedSearchCV
        else:
            self._find_best_hyperparameters_col(X=X, col=col)

        # Get the column data
        Xcol = X.loc[:, col_name]

        # Get the missing values
        missing_values = Xcol[self._missing_mask().iloc[:, col_idx]]

        # Get the features and target
        X_features = X.drop(columns=[col_name])
        X_features_new = X_features.copy()

        # Impute missing values in features
        for col in X_features.columns.tolist():
            X_features_new.loc[:, col_name] = self.best_models[col_name].predict(
                X_features
            )
            X_features_new.loc[~missing_values, col_name] = X_features.loc[
                ~missing_values, col_name
            ]

        # Return the imputed values
        return X_features_new

    def fit(self, X=None, return_evaluation_df: bool = True):
        # If X is None, use self.X
        if X is None:
            X = self.bestX

        # Get the missing mask
        missing_mask = self._missing_mask(X)

        # If any columns have missing values but no hyperparameters, find the best hyperparameters
        for col in X.columns.tolist():
            # Get the column name and index
            if isinstance(col, str):
                col_name = col
                col_idx = X.columns.tolist().index(col_name)
            else:
                col_name = X.columns.tolist()[col]
                col_idx = col

            if missing_mask[col_name].any() & (
                col not in self.best_hyperparameters.keys()
            ):
                print(f"Finding hyperparameters for column '{col}'...")
                self._find_best_hyperparameters(X=X, col=col)

        # Begin the iteration loop
        for _ in tqdm(range(self.n_iter), desc="Fitting model..."):
            iteration_change = []
            # Take the current values of X
            for c in X.columns.tolist():
                # Get the column name and index
                if isinstance(c, str):
                    col_name = c
                    col_idx = X.columns.tolist().index(col_name)
                else:
                    col_name = X.columns.tolist()[c]
                    col_idx = c

                # Refit the model for this column
                self._fit_model(target_col=col_name)

                X_cur = self.bestX[col_name]

                # Estimate missing values with the current best model
                X_new = self._predict_missing_values(X=X, col=col_name)

                # Update the current values of X, by the learning rate * (new - current)
                iteration_change.append(np.abs(self.learning_rate * (X_new - X_cur)))
                X_cur = X_cur + self.learning_rate * (X_new - X_cur)

                # Store the iteration results
                self.bestX[col_name] = X_cur
                X = self.bestX

            # Once the process refits each column once each, build eval df if needed:
            # if return_evaluation_df:
            #     eval_df = self.evaluate(X_complete=X.loc[X_train.index.to_series()], X_imputed=self.bestX)
            #     self.bestX = self._median_impute(self.bestX)
            #     self.bestX = self._mode_impute(self.bestX)
            #     yield eval_df

            # then check for convergence
            if self._check_convergence(iteration_change):
                break

    def _check_convergence(self, change):
        """
        Checks if the algorithm has converged.
        """
        # Get the absolute difference between the current and new values
        diff = pd.Series(change)

        # If the maximum difference is less than the tolerance, return True
        if diff.max().max() < self.tol:
            return True
        else:
            return False

    def transform(self, X_new):
        # 1. Get the missing mask
        # missing_mask = self._missing_mask(X_new)

        # 2. initialize imputation with median/mode
        X_new = self._median_impute(X_new)
        X_new = self._mode_impute(X_new)

        # 3. predict missing values, using the same iteration procedure as before
        for _ in tqdm(range(self.n_iter)):
            iteration_change = []
            # Take the current values of X
            for c in X_new.columns.tolist():
                X_cur = X_new[c]

                # Estimate missing values with the current best model
                X_new = self._predict_missing_values(X=X_new, col=c)

                # Update the current values of X, by the learning rate * (new - current)
                iteration_change.append(np.abs(self.learning_rate * (X_new - X_cur)))
                X_cur = X_cur + self.learning_rate * (X_new - X_cur)

                # Store the iteration results
                X_new[c] = X_cur

            # Once the process refits each column once each, check for convergence
            if self._check_convergence(iteration_change):
                break

        # 4. return the imputed values
        return X_new

    def fit_transform(self, X=None):
        """
        The fit method already transforms the data (in place - this is an iterative algorithm).
        So this method is included for convenience. It aliases the fit and transform methods.
        """
        self.fit(X)
        return self.bestX

    def _save_fitted_model(self, col):
        """
        Saves the fitted model to a file.
        """
        # Get the column name and index
        if isinstance(col, str):
            col_name = col
        else:
            col_name = self.bestX.columns.tolist()[col]

        # Save the model
        onnx_model = to_onnx(
            self.best_models[col_name], X=self.bestX.drop(columns=[col_name])
        )
        with open(f"{col_name}.onnx", "wb") as f:
            f.write(onnx_model.SerializeToString())
