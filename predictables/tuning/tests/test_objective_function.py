# FILEPATH: /home/aweaver/work/predictables/PredicTables/tuning/tests/test_objective_function.py

import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, roc_auc_score

from predictables.tuning.src._objective_function import objective_function


# Create a fixture for the model class
@pytest.fixture
def model_class():
    return RandomForestClassifier


# Create a fixture for the evaluation metric
@pytest.fixture(params=[accuracy_score, roc_auc_score, [accuracy_score, roc_auc_score]])
def evaluation_metric(request):
    return request.param


# Create a fixture for the feature matrix and target vector
@pytest.fixture
def data():
    X, y = make_classification(
        n_samples=100,
        n_features=20,
        n_informative=2,
        n_redundant=10,
        random_state=42,
    )
    X = pd.DataFrame(X)
    y = pd.Series(y)
    return X, y


# Create a fixture for the parameters
@pytest.fixture
def params():
    return {"n_estimators": 10, "max_depth": 5}


# Test the objective function
def test_objective_function(model_class, evaluation_metric, data, params):
    X, y = data
    score = objective_function(params, model_class, evaluation_metric, X, y)
    assert isinstance(score, float) or isinstance(
        score, int
    ), f"The objective function should return a float or an integer:\nexpected: float or int\nactual: {type(score)}"


def test_single_string(model_class, data, params):
    X, y = data
    score = objective_function(params, model_class, "accuracy", X, y)
    assert isinstance(score, float) or isinstance(
        score, int
    ), f"The objective function should return a float or an integer:\nexpected: float or int\nactual: {type(score)}"


def test_single_callable(model_class, data, params):
    X, y = data
    score = objective_function(params, model_class, accuracy_score, X, y)
    assert isinstance(score, float) or isinstance(
        score, int
    ), f"The objective function should return a float or an integer:\nexpected: float or int\nactual: {type(score)}"


def test_list_of_strings(model_class, data, params):
    X, y = data
    score = objective_function(params, model_class, ["accuracy", "roc_auc"], X, y)
    assert isinstance(score, float) or isinstance(
        score, int
    ), f"The objective function should return a float or an integer:\nexpected: float or int\nactual: {type(score)}"


def test_list_of_callables(model_class, data, params):
    X, y = data
    score = objective_function(
        params, model_class, [accuracy_score, roc_auc_score], X, y
    )
    assert isinstance(score, float) or isinstance(
        score, int
    ), f"The objective function should return a float or an integer:\nexpected: float or int\nactual: {type(score)}"


def test_single_string_metric(data):
    X, y = data
    params = {"n_estimators": 10}
    score = objective_function(params, RandomForestClassifier, "accuracy", X, y)
    assert isinstance(score, float) or isinstance(
        score, int
    ), f"The objective function should return a float or an integer:\nexpected: float or int\nactual: {type(score)}"


def test_single_callable_metric(data):
    X, y = data
    params = {"n_estimators": 10}
    score = objective_function(params, RandomForestClassifier, accuracy_score, X, y)
    assert isinstance(score, float) or isinstance(
        score, int
    ), f"The objective function should return a float or an integer:\nexpected: float or int\nactual: {type(score)}"


def test_multiple_metrics(data):
    X, y = data
    params = {"n_estimators": 10}
    metrics = ["accuracy", precision_score]
    score = objective_function(params, RandomForestClassifier, metrics, X, y)
    assert isinstance(score, float) or isinstance(
        score, int
    ), f"The objective function should return a float or an integer:\nexpected: float or int\nactual: {type(score)}"


def test_invalid_metric(data):
    X, y = data
    params = {"n_estimators": 10}
    with pytest.raises(ValueError):
        (
            objective_function(params, RandomForestClassifier, "invalid_metric", X, y),
            "No value error was raised...",
        )
