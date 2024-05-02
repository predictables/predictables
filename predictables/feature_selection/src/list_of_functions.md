```python
def initialize_feature_set(
    X: pd.DataFrame | pl.DataFrame | pl.LazyFrame
) -> list[str]:
    """Return a list of feature names from the input
    DataFrame.
    """
```

```python
def calculate_all_feature_correlations(
    X: pd.DataFrame | pl.DataFrame | pl.LazyFrame,
) -> pd.DataFrame | pl.DataFrame:
    """Return the correlation matrix of all features
    in the input DataFrame.
    """
```

```python
def identify_highly_correlated_pairs(
    correlations: pd.DataFrame, threshold: float = 0.5
) -> list[tuple[str, str]]:
    """Return a list of pairs of feature names with
    a correlation coefficient above the threshold.
    """
```

```python
def generate_X_y(
    X: pd.DataFrame,
    y: pd.Series,
    start_fold: int = 5,
    end_fold: int = 9
) -> tuple[
    pd.DataFrame,
    pd.Series,
    pd.DataFrame,
    pd.Series
    ]:
    """Return a generator that yields the training and
    test sets for each fold.

    Assumes a 'fold' column in X which contains the fold
    number for each row.

    Note that this is a time-series cross-validation
    generator, which means that the training set for each
    fold includes all data currently available, and the
    test set is the data in the next fold.

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
```

```python
def evaluate_feature_removal_impact(
    X: pd.DataFrame,
    y: pd.Series,
    model: SKClassifier,
    feature: str,
    start_fold: int = 5,
    end_fold: int = 5,
) -> tuple[list[float]]:
    """Evaluate the impact on model performance when a
    feature is removed using time-series cross-validation.

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
        The cross-validated evaluations of the model
        performance with and without the feature.
    """
```

```python
def select_feature_to_remove(
    score_with_1: list[float],
    score_without_1: list[float],
    feature1: str,
    score_with_2: list[float],
    score_without_2: list[float],
    feature2: str,
    tolerance: float = 1e-2,
) -> str | None:
    """Select which feature to remove based on statistical
    significance of impact scores.

    A feature is considered for removal if the mean
    improvement when it is excluded exceeds
    one standard deviation of the scores when it is
    included plus a tolerance threshold.

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
        The tolerance for considering a difference as
        statistically significant.
        Default is 1e-2.

    Returns
    -------
    str
        The name of the feature to remove, or None if
        neither meets the criterion.
    """
```