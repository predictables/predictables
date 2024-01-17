import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, log_loss, roc_auc_score

from predictables.model_analysis import calculate_error_metrics

# Mock data for testing
X = np.array([[1, 2], [3, 4], [5, 6]])
y = np.array([0, 1, 1])


# Fixture to create a model with perfect predictions
@pytest.fixture
def create_perfect_model():
    model = RandomForestClassifier()
    model.fit(X, y)
    return model


# Fixture to create a model with poor predictions
@pytest.fixture
def create_poor_model():
    model = RandomForestClassifier()
    # Training with wrong targets to ensure poor performance
    wrong_y = np.array([1, 0, 0])
    model.fit(X, wrong_y)
    return model


# Test with valid data and a perfectly predicting model
def test_with_perfect_prediction(create_perfect_model):
    model = create_perfect_model
    metrics = calculate_error_metrics(X, y, model)

    assert (
        metrics["accuracy"] == 1
    ), f"Accuracy should be 1 for a perfect model, but is {metrics['accuracy']}"
    assert (
        metrics["precision"] == 1
    ), f"Precision should be 1 for a perfect model, but is {metrics['precision']}"
    assert (
        metrics["recall"] == 1
    ), f"Recall should be 1 for a perfect model, but is {metrics['recall']}"
    assert (
        metrics["f1_score"] == 1
    ), f"F1 score should be 1 for a perfect model, but is {metrics['f1_score']}"
    assert np.array_equal(
        metrics["confusion_matrix"], confusion_matrix(y, model.predict(X)).tolist()
    ), f"Confusion matrix should be {confusion_matrix(y, model.predict(X)).tolist()} for a perfect model, but is {metrics['confusion_matrix']}"


# Test with valid data and a poorly predicting model
def test_with_poor_prediction(create_poor_model):
    model = create_poor_model
    metrics = calculate_error_metrics(X, y, model)

    assert (
        metrics["accuracy"] < 1
    ), f"Accuracy should be less than 1 for a poor model, but is {metrics['accuracy']}"
    assert (
        metrics["precision"] < 1
    ), f"Precision should be less than 1 for a poor model, but is {metrics['precision']}"
    assert (
        metrics["recall"] < 1
    ), f"Recall should be less than 1 for a poor model, but is {metrics['recall']}"
    assert (
        metrics["f1_score"] < 1
    ), f"F1 score should be less than 1 for a poor model, but is {metrics['f1_score']}"


# Test ROC-AUC and Log Loss for a binary classification model
def test_roc_auc_and_log_loss(create_perfect_model):
    model = create_perfect_model
    metrics = calculate_error_metrics(X, y, model)

    if "roc_auc" in metrics:
        assert metrics["roc_auc"] == roc_auc_score(
            y, model.predict_proba(X)[:, 1]
        ), f"ROC-AUC should be {roc_auc_score(y, model.predict_proba(X)[:, 1])} for a binary classification model, but is {metrics['roc_auc']}"

    if "log_loss" in metrics:
        assert metrics["log_loss"] == log_loss(
            y, model.predict_proba(X)[:, 1]
        ), f"Log Loss should be {log_loss(y, model.predict_proba(X)[:, 1])} for a binary classification model, but is {metrics['log_loss']}"
