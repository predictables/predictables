from typing import Union, Dict, List
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer
import numpy as np
from sklearn.base import BaseEstimator
from PredicTables.util import harmonic_mean


def objective_function(
    params: Dict[str, Union[int, float]],
    model_class,
    evaluation_metric: Union[str, List[str]],
) -> float:
    """
    Fits a generic model (implementing the sklearn API) with the given parameters,
    and returns a given evaluation metric.

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
    else:
        scorers = [make_scorer(metric) for metric in evaluation_metric]
        scorer = make_scorer(hmean, scorers=scorers)

    # Evaluate the model using cross-validation
    scores = cross_val_score(
        model, X_train, y_train, cv=5, scoring="neg_mean_squared_error"
    )

    # Return the negative mean of the scores (since we want to minimize the metric)
    return -scores.mean()
