from dataclasses import dataclass
from typing import Union

import pandas as pd
from joblib import Parallel, delayed

from PredicTables.impute.src.impute_with_trained_model import impute_single_column
from PredicTables.impute.src.initial_impute import initial_impute
from PredicTables.impute.src.train_catboost_model import train_one_catboost_model


@dataclass
class ModelImputer:
    df: pd.DataFrame = None
    missing_mask: Union[pd.DataFrame, str] = None
    n_models: int = 0
    initial_impute: bool = True

    def _initial_impute(self) -> None:
        if self.initial_impute:
            self.imputed_df = initial_impute(self.df).collect().to_pandas()
            # Update the dtype to equal that of the original df
            for col in self.imputed_df:
                self.imputed_df[col] = self.imputed_df[col].astype(self.df[col].dtype)
        else:
            self.imputed_df = self.df.copy()

    def __post_init__(self):
        # Read in the missing mask if it is a string
        if isinstance(self.missing_mask, str):
            s = self.missing_mask.copy()
            self.missing_mask = pd.read_parquet(s)
            if "sort_order" in self.missing_mask.columns.tolist():
                self.missing_mask.drop(columns=["sort_order"], inplace=True)
            if "date_order" in self.missing_mask.columns.tolist():
                self.missing_mask.drop(columns=["date_order"], inplace=True)

        # Check that the missing mask is a dataframe
        assert isinstance(
            self.missing_mask, pd.DataFrame
        ), f"missing_mask must be a pandas DataFrame, not {type(self.missing_mask)}"

        # Set the columns and missing columns
        self.columns = self.missing_mask.columns.tolist()
        missing_columns = self.missing_mask.sum()
        self.missing_columns = [
            # include only columns with at least one missing value
            c
            for c in missing_columns.index.tolist()
            if (missing_columns[c] > 0)
        ]

        # Record the categorical columns for fitting catboost models
        self.categorical_columns = []

        # Check that the missing mask and df have the same columns
        # If not, shrink the larger one to match the smaller one
        if self.missing_mask.shape[1] > self.df.shape[1]:
            self.missing_mask = self.missing_mask[self.df.columns.tolist()]
        elif self.missing_mask.shape[1] < self.df.shape[1]:
            self.df = self.df[self.missing_mask.columns.tolist()]

        # Initialze the model attributes to None -- they will become fitted models
        # when fit_models is called
        for missing_col in self.missing_columns:
            setattr(self, missing_col, None)

        # Loop over the columns and set the dtypes
        for col in self.columns:
            # Set the categorical columns
            if isinstance(self.df[col].dtype, pd.CategoricalDtype):
                # Make sure it is categorical, then add to categorical_columns
                self.df[col] = self.df[col].astype("category")
                self.categorical_columns.append(col)

            elif isinstance(self.df[col].dtype, pd.StringDtype):
                # Recode to categorical, then add to categorical_columns
                self.df[col] = self.df[col].astype("category")
                self.categorical_columns.append(col)

            elif pd.api.types.is_object_dtype(self.df[col]):
                # Recode to categorical, then add to categorical_columns
                self.df[col] = self.df[col].astype("category")
                self.categorical_columns.append(col)

            elif self.df[col].dtype in [
                "float64",
                "float32",
                "int64",
                "int32",
                "int16",
                "int8",
                "uint64",
                "uint32",
                "uint16",
                "uint8",
            ]:
                # Cast all remaining numeric columns to float64 for consistency
                self.df[col] = self.df[col].astype("float64")

            elif isinstance(self.df[col].dtype, pd.DatetimeTZDtype):
                raise ValueError(
                    f"Column {col} is a datetime column, and should have been removed by this point."
                )

            elif isinstance(self.df[col].dtype, pd.BooleanDtype):
                raise ValueError(
                    f"""Column {col} is a boolean column, and should have been removed or converted by this point. It is worth considering convreting to a categorical column, coded as '1' and '0' for True and False, respectively. Most binary columns are coded this way, and it is simpler than using a boolean column, because there are fewer cases to keep track of."""
                )

            else:
                raise ValueError(
                    f"""Column {col} has an unrecognized dtype: {self.df[col].dtype}. Please convert to a recognized dtype (categorical or numeric) before imputing."""
                )

        # Update the imputed_df columns (if initial_impute is True)
        if self.initial_impute:
            self._initial_impute()
        self.imputed_df0 = self.imputed_df.copy()

    def __repr__(self) -> str:
        return f"models(n={self.n_models})"

    def __str__(self) -> str:
        return f"models(n={self.n_models})"

    def _fit_model(self, col: str) -> None:
        """
        Fit a single CatBoost model to a single column. This is a helper method
        and is not intended to be called directly. The .fit_models() method calls
        this function in parallel, and is generally preferred.

        Parameters
        ----------
        col : str
            The name of the column to fit a model to.

        Returns
        -------
        None, but sets the class attribute to the fitted model.
        """
        # Fit a model to the column if it is initialized - eg if there is a class attribute
        # with the same name as the column

        # Check that there is a class attribute with the same name as the column
        if hasattr(self, col):
            # If so, fit a model to the column
            model = train_one_catboost_model(
                self.imputed_df, target_column=col, cv_folds=None
            )
            # Set the class attribute to the model
            setattr(self, col, model)

    def fit_models(
        self,
        n_jobs: int = 4,
        backend: str = "multiprocessing",
        verbose: Union[bool, int] = False,
    ) -> None:
        """
        Fit CatBoost models to all columns with missing values. This method
        calls the _fit_model() method in parallel, and is generally preferred
        to calling _fit_model() directly.

        Parameters
        ----------
        n_jobs : int, optional
            The number of jobs to run in parallel. The default is 4.
        backend : str, optional
            The backend to use for parallelization. The default is "multiprocessing".
        verbose : Union[bool, int], optional
            Whether to print progress. The default is False, but can select up to a
            verbosity of 10, which provides some progress bar information for each
            worker.

        Returns
        -------
        None, but sets the class attributes to the fitted models from the parallel
        process.
        """
        # Get the function to call in parallel
        f = getattr(self, "_fit_model")

        # Call the function in parallel
        Parallel(
            n_jobs=n_jobs,  # How many cores to use
            verbose=verbose,  # Whether to print progress
            backend=backend,  # Which backend to use for parallelization
        )(
            # Call the function in parallel for each column with
            # missing values, setting the class attribute to the
            # fitted model in each case
            delayed(f)(col)
            for col in self.missing_columns
        )

    def _impute_model(self, col: str, learning_rate: float) -> pd.Series:
        """
        Impute a single column using a trained CatBoost model. This is a helper
        method and is not intended to be called directly. The .impute_models()
        method calls this function in parallel, and is generally preferred.

        Parameters
        ----------
        col : str
            The name of the column to impute.
        learning_rate : float
            The learning rate to use for the imputation. The
            imputed value is a weighted average of the current value and the
            predicted value from the model, where the weight on the predicted
            value is the learning rate.

        Returns
        -------
        pd.Series
            A series with the imputed values.
        """

        if hasattr(self, col):
            # Impute the column if there is a class attribute with the same name
            # as the column -- this means that there is a fitted model for the column
            updated_df = impute_single_column(
                df=self.imputed_df,  # Start from current imputed df
                missing_mask=self.missing_mask,  # Only impute missing values
                column=col,
                trained_model=getattr(self, col)
                if hasattr(self, col)
                else print(
                    "No model fit for this column"
                ),  # Get the model from the class attribute if it exists
                learning_rate=learning_rate,
                only_missing=True,  # Only impute missing values
                cv_fold=None,  # Don't use a cross-validation fold
            )
        else:
            # If there is no class attribute with the same name as the column,
            # then there is no model fit for the column, so return the original
            # column and move on
            updated_df = self.imputed_df

        # Return the updated column (or original if no model fit)
        return updated_df[col]

    def impute_models(
        self,
        learning_rate: float = 0.5,
        n_jobs: int = 4,
        backend: str = "multiprocessing",
        verbose: bool = False,
    ) -> None:
        """
        Impute all columns with missing values using the fitted CatBoost models.
        This method calls the _impute_model() method in parallel, and is generally
        preferred to calling _impute_model() directly.

        Parameters
        ----------
        learning_rate : float, optional
            The learning rate to use for the imputation. The default is 0.5. The
            imputed value is a weighted average of the current value and the
            predicted value from the model, where the weight on the predicted
            value is the learning rate.
        n_jobs : int, optional
            The number of jobs to run in parallel. The default is 4.
        backend : str, optional
            The backend to use for parallelization. The default is "multiprocessing".
        verbose : bool, optional
            Whether to print progress. The default is False.

        Returns
        -------
        None, but sets the class attributes to the imputed values from the parallel
        process.
        """
        # Set the current imputed df to the original imputed df
        current_df = self.imputed_df.copy()

        # Get the function to call in parallel
        f = getattr(self, "_impute_model")

        # Call the function in parallel and get the updated columns
        updated_cols = Parallel(
            n_jobs=n_jobs,  # How many cores to use
            verbose=verbose,  # Whether to print progress
            backend=backend,  # Which backend to use for parallelization
        )(
            # Call the function in parallel for each column with
            # missing values, setting the class attribute to the
            # imputed values in each case
            delayed(f)(col, learning_rate)
            for col in self.missing_columns
        )

        # Add the updated columns to an updated df
        updated_df = current_df.copy()
        for col in self.missing_columns:
            updated_df[col] = updated_cols.pop(0)

        # Update the imputed_df with the weighted average of the
        # current and updated dfs
        self.imputed_df = (1 - learning_rate) * current_df + learning_rate * updated_df
        self.imputed_df = (1 - learning_rate) * current_df + learning_rate * updated_df
