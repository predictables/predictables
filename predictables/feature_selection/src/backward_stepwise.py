"""Perform backward stepwise feature selection by recursively eliminating features from a model based on their correlation and impact on predictive performance, evaluated through AUC metrics.

The process involves:
1. Identifying pairs of features with a Pearson correlation coefficient above a specified threshold (default is 0.5).
2. For each correlated pair, the model's performance is evaluated three times: using all features, minus the first feature, and minus the second feature.
3. The feature whose removal causes the least decrease in AUC (within one standard deviation of the AUC when all features are used) is eliminated. This standard deviation acts as a threshold to ensure that only features whose absence does not statistically degrade the model's performance are removed.
4. This iterative process continues until all pairs of correlated features have been evaluated, refining the feature set to improve model simplicity without significantly reducing performance.

Parameters
----------
model : SKClassifier
    The initialized model object whose features are to be evaluated. The model must implement the fit and predict methods.
threshold : float
    The correlation threshold above which feature pairs are considered for elimination.

Returns
-------
pl.LazyFrame
    The dataset with the reduced set of features after removing highly correlated and less impactful ones.
"""

from __future__ import annotations
import polars as pl
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from predictables.util import (
    SKClassifier,
)  # duck type indicating that the model is an sklearn classifier implementing the fit and predict methods


def compute_feature_correlations(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Compute the correlation between all features in the dataset.

    Loops over the columns in the dataset and computes the correlation
    between each pair of columns. The correlation is computed using the
    Pearson correlation coefficient. Returns a LazyFrame with the
    columns 'col1', 'col2', and 'correlation' representing the two
    columns being compared and the correlation between them, respectively,
    sorted in from most to least correlated.

    Parameters
    ----------
    lf : pl.LazyFrame
        The dataset to compute feature correlations for.

    Returns
    -------
    pl.LazyFrame
        A LazyFrame with the columns 'col1', 'col2', and 'correlation' representing the
        two columns being compared and the correlation between them, respectively, sorted
        in descending order of correlation.
    """
    cols = lf.columns
    return (
        lf.select(
            [
                # Correlation between all pairs of columns i != j
                pl.corr(cols[i], cols[j]).alias(f"{cols[i]}_corr_{cols[j]}")
                for i in range(len(cols))
                for j in range(i + 1, len(cols))
            ]
        )
        .collect()
        # We have a column for each pair of columns, so we need
        # to transpose the data to get an iterable of (col1, col2, correlation)
        .transpose(include_header=True)
        .lazy()
        .with_columns(
            [
                # Split the column name to get the two column names
                pl.col("column").str.split("_corr_").list.get(0).alias("col1"),
                pl.col("column").str.split("_corr_").list.get(1).alias("col2"),
                pl.col("column_0").abs().alias("correlation"),
            ]
        )
        .drop("column_0")
        .drop("column")
        .filter(pl.col("col1") != pl.col("col2"))
        # Sort the correlations in descending order to make it easier to
        # identify highly correlated pairs
        .sort("correlation", descending=True)
    )


def identify_highly_correlated_pairs(
    lf: pl.LazyFrame, threshold: float = 0.5
) -> list[tuple[str, str]]:
    """Return highly-correlated pairs of columns.

    Here 'highly-correlated' means that the absolute value of their
    Pearson correlation coefficient is > 0.5. While this might seem
    like a low threshold, keep in mind that each pair of columns will be
    emprically tested, so the decision was made to keep the threshold
    fairly low to ensure that we don't bother with obviously weakly-
    correlated pairs, but also ensure that all marginal cases are
    actually tested.
    """
    corr = (
        compute_feature_correlations(lf)
        .filter(pl.col("correlation") > threshold)
        .select([pl.col("col1"), pl.col("col2")])
        .collect()
        .to_pandas()
    )

    return corr.apply(lambda x: (x["col1"], x["col2"]), axis=1).tolist()


def generate_X_y(
    X: pl.LazyFrame, y: np.ndarray
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray]:
    """Yield training and testing splits for time series cross validation.

    The training set consists of all observations up to the current fold, while the testing set consists of the current fold.
    The folds are determined prior to calling this function, and are determined by the submission recieved month:
    - The test set contains the two most recent months
    - The validation set contains the two months prior to that
    - The training set numbers the next 10 months in reverse order, with all months prior to 10 months ago given the fold label
      of 0.

    Parameters
    ----------
    X : pl.LazyFrame
        The dataset to split into training and testing sets.
    y : np.ndarray
        The target variable to split into training and testing sets.

    Yields
    ------
    tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray]
        The training and testing splits of the dataset and target variable.
    """
    for i in range(5, 11):
        train_idx = (
            X.with_row_index()
            .filter(pl.col("fold") < i + 1)
            .select(pl.col("index"))
            .collect()
            .to_numpy()
        )

        val_idx = (
            X.with_row_index()
            .filter(pl.col("fold") == i + 1)
            .select(pl.col("index"))
            .collect()
            .to_numpy()
        )

        dfx = X.collect().to_pandas()

        yield dfx.iloc[train_idx], y[train_idx], dfx.iloc[val_idx], y[val_idx]


def fit_models(
    model: SKClassifier, col1: str, col2: str, columns_to_drop: list[str]
) -> tuple[list[SKClassifier], list[SKClassifier], list[SKClassifier]]:
    """Fit the current model and two models with one of the correlated columns removed."""
    gen_cur, gen1, gen2 = generate_X_y(), generate_X_y(), generate_X_y()

    cur_models = [
        model.fit(
            X_train.drop(columns_to_drop).collect().to_numpy(),
            y_train.to_numpy().ravel(),
        )
        for X_train, y_train, _, _ in gen_cur
    ]

    ex1_models = [
        model.fit(
            X_train.drop(columns_to_drop).drop(col1).collect().to_numpy(),
            y_train.to_numpy().ravel(),
        )
        for X_train, y_train, _, _ in gen1
    ]

    ex2_models = [
        model.fit(
            X_train.drop(columns_to_drop).drop(col2).collect().to_numpy(),
            y_train.to_numpy().ravel(),
        )
        for X_train, y_train, _, _ in gen2
    ]

    return cur_models, ex1_models, ex2_models


def calculate_aucs_from_fitted_models(
    model: SKClassifier, col1: str, col2: str, columns_to_drop: list[str]
) -> tuple[list[float], list[float], list[float]]:
    """Calculate the AUC of the current model and two models with one of the correlated columns removed."""
    cur_model, ex1_model, ex2_model = fit_models(model, col1, col2, columns_to_drop)
    return (
        [roc_auc_eval(model, X_test, y_test) for X_test, y_test, _, _ in cur_model],
        [
            roc_auc_eval(model, X_test, y_test, [col1])
            for X_test, y_test, _, _ in ex1_model
        ],
        [
            roc_auc_eval(model, X_test, y_test, [col2])
            for X_test, y_test, _, _ in ex2_model
        ],
    )


def evaluate_what_if_any_column_to_drop(
    current_auc: list[float], ex1_auc: list[float], ex2_auc: list[float]
) -> int:
    """Return 0, 1, or 2 to indicate you should drop no column, column 1, or column 2, respectively.

    If the mean AUC from the model fit without either column 1 or 2 is greater than the mean AUC
    of the current model less one standard deviation, make the new current
    model be the one with the highest mean AUC by returning either 1 or 2 depending on which was
    higher. If both smaller models are worse than the current, do not make any adjustment to the
    current model and continue to the next pair-return 0 to indicate this.
    """
    # Test whichever column has the highest auc
    if max(np.mean(ex1_auc), np.mean(ex2_auc)) > np.mean(current_auc) - np.std(
        current_auc
    ):
        return 1 if np.mean(ex1_auc) < np.mean(ex2_auc) else 2
    else:
        return 0


def drop_column_from_data(col_to_drop: int, col1: str, col2: str) -> str:
    """Drop the column from the dataset."""
    return col1 if col_to_drop == 1 else col2


def backward_stepwise_feature_selection(
    X: pl.LazyFrame, y: np.ndarray, model: SKClassifier, threshold: float = 0.5
) -> pl.LazyFrame:
    """Select a subset of the current features by iteratively removing highly-correlated features that do not significantly impact the model.

    The process involves:
    1. Identifying pairs of features with a Pearson correlation coefficient above a specified threshold (default is 0.5).
    2. For each correlated pair, the model's performance is evaluated three times: using all features, minus the first feature, and minus the second feature.
    3. The feature whose removal causes the least decrease in AUC (within one standard deviation of the AUC when all features are used) is eliminated. This standard deviation acts as a threshold to ensure that only features whose absence does not statistically degrade the model's performance are removed.
    4. This iterative process continues until all pairs of correlated features have been evaluated, refining the feature set to improve model simplicity without significantly reducing performance.

    Parameters
    ----------
    X : pl.LazyFrame
        The dataset to reduce by removing highly correlated features.
    y : np.ndarray
        The target variable to use for model evaluation.
    model : SKClassifier
        The initialized model object whose features are to be evaluated. The model must implement the fit and predict methods.
    threshold : float
        The correlation threshold above which feature pairs are considered for elimination. Default is 0.5.

    Returns
    -------
    pl.LazyFrame
        The dataset with the reduced set of features after removing highly correlated and less impactful ones.
    """
    gen = generate_X_y(X, y)
    X_train, _, _, _ = next(gen)

    columns_to_drop = set()

    corr_pairs = identify_highly_correlated_pairs(X_train, threshold)

    for col1, col2 in corr_pairs:
        cur, ex1, ex2 = calculate_aucs(model, col1, col2, columns_to_drop)

        col_to_drop = evaluate_what_if_any_column_to_drop(cur, ex1, ex2)

        if col_to_drop > 0:
            columns_to_drop.append(drop_column_from_data(col_to_drop, col1, col2))

    return X_train.drop(columns_to_drop)


def evaluate_single_model(
    model: SKClassifier, col1: str, col2: str, columns_to_drop: list[str]
) -> tuple[float, float, float]:
    """Evaluate the AUC of the current model and two models with one of the correlated columns removed."""
    current_model, model_ex_1, model_ex_2 = fit_models(
        model, col1, col2, columns_to_drop
    )
    return roc_auc_eval(current_model, model_ex_1, model_ex_2, col1, col2)


def roc_auc_eval(
    model: SKClassifier,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    drop_columns: list[str] | None = None,
) -> float:
    """Evaluate the AUC score of a model on the test set optionally dropping specified columns from the test set."""
    if drop_columns is not None:
        X_test = X_test.drop(columns=drop_columns)
    predictions = model.predict_proba(X_test)[:, 1]  # Assuming binary classification
    return roc_auc_score(y_test, predictions)
