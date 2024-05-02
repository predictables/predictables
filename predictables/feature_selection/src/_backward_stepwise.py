from __future__ import annotations

import pandas as pd
import polars as pl
import numpy as np
from sklearn.base import clone
from sklearn.metrics import roc_auc_score

from predictables.util import to_pd_df, SKClassifier


def initialize_feature_set(X: pd.DataFrame | pl.DataFrame | pl.LazyFrame) -> list[str]:
    """Return a list of feature names from the input DataFrame."""
    return list(X.columns) if isinstance(X, pd.DataFrame) else X.columns


def calculate_all_feature_correlations(
    X: pd.DataFrame | pl.DataFrame | pl.LazyFrame,
) -> pd.DataFrame | pl.DataFrame:
    """Return the correlation matrix of all features in the input DataFrame."""
    return to_pd_df(X).corr()


def identify_highly_correlated_pairs(
    correlations: pd.DataFrame, threshold: float = 0.5
) -> list[tuple[str, str]]:
    """Return a list of pairs of feature names with a correlation coefficient above the threshold."""
    return [
        (correlations.columns[i], correlations.columns[j])
        for i in range(correlations.shape[0])
        for j in range(i + 1, correlations.shape[1])
        if abs(correlations.iloc[i, j]) > threshold
    ]


def generate_X_y(
    X: pd.DataFrame, y: pd.Series, start_fold: int = 5, end_fold: int = 9
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Return a generator that yields the training and test sets for each fold.

    Assumes a 'fold' column in X which contains the fold number for each row.

    Note that this is a time-series cross-validation generator, which means that the training set for each fold includes all data currently available, and the test set is the data in the next fold.

    Parameters
    ----------
    X : pd.DataFrame
        The input features.
    y : pd.Series
        The target variable.
    start_fold : int
        The starting fold number.
    end_fold : int
        The ending fold number.

    Yields
    ------
    X_train : pd.DataFrame
        The training features.
    y_train : pd.Series
        The training target variable.
    X_test : pd.DataFrame
        The test features.
    y_test : pd.Series
        The test target variable.
    """
    if "fold" not in X.columns:
        raise ValueError("DataFrame must contain a 'fold' column.")

    for f in range(start_fold, end_fold + 1):
        train_idx = X[X["fold"] < f].index
        test_idx = X[X["fold"] == f].index
        if len(test_idx) == 0:
            raise ValueError(f"No test data available for fold {f}.")
        yield X.loc[train_idx], y.loc[train_idx], X.loc[test_idx], y.loc[test_idx]


def evaluate_feature_removal_impact(
    X: pd.DataFrame,
    y: pd.Series,
    model: SKClassifier,
    feature: str,
    start_fold: int = 5,
    end_fold: int = 5,
) -> tuple[list[float]]:
    """Evaluate the impact on model performance when a feature is removed using time-series cross-validation.

    Parameters
    ----------
    X : pd.DataFrame
        The input features.
    y : pd.Series
        The target variable.
    model : SKClassifier
        An sklearn-style classifier.
    feature : str
        The feature to consider for removal.
    start_fold : int
        The starting fold for time-series cross-validation.
    end_fold : int
        The ending fold for time-series cross-validation.

    Returns
    -------
    tuple[list[float]]
        The cross-validated evaluations of the model performance with and without the feature.
    """
    score_with, score_without = [], []
    generator = generate_X_y(X, y, start_fold, end_fold)

    for X_train, y_train, X_test, y_test in generator:
        # Train and evaluate with the feature
        model_clone = clone(model)
        model_clone.fit(X_train, y_train)
        y_pred = model_clone.predict_proba(X_test)[..., 1]
        score_with.append(roc_auc_score(y_test, y_pred))

        # Train and evaluate without the feature
        X_train_reduced = X_train.drop(columns=[feature])
        X_test_reduced = X_test.drop(columns=[feature])
        model_clone.fit(X_train_reduced, y_train)
        y_pred = model_clone.predict_proba(X_test_reduced)[..., 1]
        score_without.append(roc_auc_score(y_test, y_pred))

    # Return the list of performances
    return score_with, score_without


def select_feature_to_remove(
    score_with_1: list[float],
    score_without_1: list[float],
    feature1: str,
    score_with_2: list[float],
    score_without_2: list[float],
    feature2: str,
    tolerance: float = 1e-2,
) -> str | None:
    """Select which feature to remove based on statistical significance of impact scores.

    A feature is considered for removal if the mean improvement when it is excluded exceeds
    one standard deviation of the scores when it is included plus a tolerance threshold.

    Parameters
    ----------
    score_with_1 : list[float]
        AUC scores with feature1 included.
    score_without_1 : list[float]
        AUC scores with feature1 excluded.
    feature1 : str
        The first feature being considered for removal.
    score_with_2 : list[float]
        AUC scores with feature2 included.
    score_without_2 : list[float]
        AUC scores with feature2 excluded.
    feature2 : str
        The second feature being considered for removal.
    tolerance : float
        The tolerance for considering a difference as statistically significant.
        Default is 1e-2.

    Returns
    -------
    str
        The name of the feature to remove, or None if neither meets the criterion.
    """
    mean_with_1 = np.mean(score_with_1)
    std_with_1 = np.std(score_with_1)
    mean_without_1 = np.mean(score_without_1)

    mean_with_2 = np.mean(score_with_2)
    std_with_2 = np.std(score_with_2)
    mean_without_2 = np.mean(score_without_2)

    # Calculate the net improvement threshold, considering one standard deviation
    net_improvement_1 = (mean_without_1 - mean_with_1) - std_with_1
    net_improvement_2 = (mean_without_2 - mean_with_2) - std_with_2

    # Determine which feature to remove based on net improvement
    if net_improvement_1 > tolerance and net_improvement_2 > tolerance:
        # Both features show net improvement when removed, remove the one with more net improvement
        return feature2 if net_improvement_1 < net_improvement_2 else feature1
    elif net_improvement_1 > tolerance:
        return feature1
    elif net_improvement_2 > tolerance:
        return feature2

    return None


def backward_stepwise_feature_selection(
    X: pd.DataFrame,
    y: pd.Series,
    model: SKClassifier,
    start_fold: int = 5,
    end_fold: int = 9,
    correlation_threshold: float = 0.75,
    tolerance: float = 1e-2,
) -> list[str]:
    """Perform backward stepwise feature selection using time-series cross-validation.

    This process recursively eliminates features that show high correlation with
    other features, and minimal impact on the predictive performance of a model
    when removed.

    Parameters
    ----------
    X : pd.DataFrame
        The input features.
    y : pd.Series
        The target variable.
    model : SKClassifier
        An sklearn-style classifier.
    start_fold : int
        The starting fold for time-series cross-validation.
    end_fold : int
        The ending fold for time-series cross-validation.
    correlation_threshold : float
        The threshold for considering features as highly correlated.
    tolerance : float
        The tolerance for considering a difference as statistically significant.
        Default is 1e-2.

    Returns
    -------
    list[str]
        The list of selected features after backward elimination.

    Raises
    ------
    ValueError
        If less than two features are provided.

    Detailed Process
    ----------------
    1. Calculate correlations between all pairs of features.
    2. Identify pairs of features with correlation above the threshold.
    3. Evaluate the impact of removing each feature in these pairs on model performance.
    4. Remove the feature whose exclusion leads to the least performance degradation.
    5. Repeat until no highly correlated pairs remain or performance cannot be maintained.
    """
    features = initialize_feature_set(X)
    if len(features) < 2:
        raise ValueError(
            "Backward stepwise feature selection requires at least two features."
        )

    correlations = calculate_all_feature_correlations(X[features])
    correlated_pairs = identify_highly_correlated_pairs(
        correlations, correlation_threshold
    )

    # Continue until no features can be justifiably removed
    while len(features) > 1:
        scores_with = {}
        scores_without = {}

        for feature in features:
            # Evaluate the impact of removing each feature
            scores_with[feature], scores_without[feature] = (
                evaluate_feature_removal_impact(
                    X[features], y, model, feature, start_fold, end_fold
                )
            )

        # Check correlation and performance impacts to select features to remove
        feature_to_remove = None
        for feature1, feature2 in correlated_pairs:
            if feature1 in features and feature2 in features:
                feature_to_remove = select_feature_to_remove(
                    scores_with[feature1],
                    scores_without[feature1],
                    feature1,
                    scores_with[feature2],
                    scores_without[feature2],
                    feature2,
                    tolerance,
                )
                if feature_to_remove:
                    break

        # If no correlated feature meets the criterion for removal, proceed by performance
        if not feature_to_remove:
            for feature in features:
                if (
                    np.mean(scores_without[feature]) - np.mean(scores_with[feature])
                    > tolerance
                ):
                    feature_to_remove = feature
                    break

        # If no feature meets the criterion for removal, stop
        if feature_to_remove is None:
            break

        # Remove the selected feature and update the correlation pairs
        features.remove(feature_to_remove)
        correlated_pairs = [
            (f1, f2)
            for f1, f2 in correlated_pairs
            if f1 != feature_to_remove and f2 != feature_to_remove
        ]
        print(f"Removed feature: {feature_to_remove}")

    return features
