from typing import Callable, Dict, List, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score

from predictables.util import harmonic_mean


def standardize_scores(scores: List[float]) -> List[float]:
    """
    Standardize a list of scores.

    Parameters
    ----------
    scores : List[float]
        List of scores to be standardized.

    Returns
    -------
    List[float]
        List of standardized scores.
    """
    mean = np.mean(scores)
    std = np.std(scores)
    out = []
    for s in scores:
        if std != 0:
            out.append((s - mean) / std)
        else:
            out.append(0)
    return out


def objective_function(
    params: Dict[str, Union[int, float]],
    model_class: BaseEstimator,
    evaluation_metric: Union[Callable, str, List[Callable], List[str]],
    X: pd.DataFrame,
    y: pd.Series,
    cv_folds: int = 5,
) -> float:
    """
    Fits a generic model (implementing the sklearn API) with the given parameters,
    and returns a given evaluation metric.

    Parameters
    ----------
    params : Dict[str, Union[int, float]]
        Parameters to be passed to the model.
    model_class : BaseEstimator
        A class implementing the sklearn API.
    evaluation_metric : Union[Callable, str, List[Callable], List[str]]
        The evaluation metric to be used.
    X : pd.DataFrame
        The feature matrix.
    y : pd.Series
        The target vector.
    cv_folds : int
        The number of cross-validation folds.

    Returns
    -------
    float
        The harmonic mean of the standardized evaluation metric scores.
    """

    model = model_class(**params)

    if not isinstance(evaluation_metric, list):
        evaluation_metric = [evaluation_metric]

    scores = []
    for metric in evaluation_metric:
        if callable(metric):
            scorer = make_scorer(metric, greater_is_better=False)
        else:
            scorer = metric

        score = cross_val_score(
            model, X, y, cv=cv_folds, scoring=scorer, n_jobs=-1, error_score="raise"
        )
        scores.append(score.mean())

    standardized_scores = standardize_scores(scores)
    return harmonic_mean(*standardized_scores)
