from typing import Dict, List, Union

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score

from PredicTables.util import harmonic_mean


def objective_function(
    params: Dict[str, Union[int, float]],
    model_class: BaseEstimator,
    evaluation_metric: Union[str, List[str]],
    X: pd.DataFrame,
    y: pd.Series,
) -> float:
    """
    Fits a generic model (implementing the sklearn API) with the given parameters,
    and returns a given evaluation metric.

    Any evaluation metric should be configured such that a smaller value is better.

    Parameters
    ----------
    params : dict
        Parameters to be passed to the model. In the form of a dictionary with
        keys as the parameter names and values as the parameter values.
    model_class : inherited from BaseEstimator
        A class implementing the sklearn API. Must implement the fit and predict
        methods for the cross-validation to work.
    evaluation_metric : str or list of str
        The evaluation metric to be used. Must be a valid sklearn metric.

        - A good default for regression problems is "neg_mean_squared_error".
        - A good default for classification problems is "accuracy".

        If a list of strings representing valid sklearn metrics is passed, the
        harmonic mean of the scores will be returned.
    X : pandas.DataFrame
        The feature matrix.
    y : pandas.Series
        The target vector.

    Returns
    -------
    float
        The evaluation metric score.
    """
    # Create a model instance
    model = model_class(**params)

    # Create a scorer instance for the evaluation metric if it is a string,
    # and for each individual metric if it is a list of strings
    if isinstance(evaluation_metric, str):
        scorer = make_scorer(evaluation_metric)
    elif isinstance(evaluation_metric, list):
        scorers = [make_scorer(metric) for metric in evaluation_metric]
        scorer = make_scorer(map(harmonic_mean, *scorers))

    # Evaluate the model using cross-validation
    scores = cross_val_score(
        model, X, y, cv=5, scoring=scorer, n_jobs=-1, error_score="raise"
    )

    # Return the negative mean of the scores (since we want to minimize the metric)
    return -scores.mean()
