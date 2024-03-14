from __future__ import annotations

from typing import Union

import pandas as pd
import polars as pl
import sklearn
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


def calculate_error_metrics(
    X: Union[pd.DataFrame, pl.DataFrame, pl.LazyFrame],
    y: Union[pd.Series, pl.Series],
    model: sklearn.base.BaseEstimator,
) -> dict:
    """
    Calculates comprehensive error metrics for the model.

    Parameters
    ----------
    X : Union[pd.DataFrame, pl.DataFrame, pl.LazyFrame]
        Input dataset of model features.
    y : Union[pd.Series, pl.Series]
        Target variable.
    model : sklearn.base.BaseEstimator
        Model to be validated. Must be a scikit-learn estimator, or an estimator that
        implements the scikit-learn API, such as xgboost, lightgbm, etc.

    Returns
    -------
    dict
        Dictionary of error metrics.
    """
    # Predicting using the model
    y_pred = model.predict(X)
    y_pred_proba = (
        model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else None
    )

    # Calculating metrics
    metrics = {
        "accuracy": accuracy_score(y, y_pred),
        "precision": precision_score(y, y_pred, average="weighted"),
        "recall": recall_score(y, y_pred, average="weighted"),
        "f1_score": f1_score(y, y_pred, average="weighted"),
        "confusion_matrix": confusion_matrix(y, y_pred).tolist(),
        "mcc": matthews_corrcoef(y, y_pred),
    }

    if y_pred_proba is not None:
        metrics["roc_auc"] = roc_auc_score(y, y_pred_proba)
        metrics["log_loss"] = log_loss(y, y_pred_proba)

    return metrics
