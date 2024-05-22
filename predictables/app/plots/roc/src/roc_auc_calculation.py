"""Calculate the ROC curve and AUC score for the ROC-AUC plot."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc
from sklearn.linear_model import LogisticRegressionCV


def calculate_roc_auc(
    train_data: pd.DataFrame,
    validation_data: pd.DataFrame,
    features: str | list[str],
    target: str,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Calculate the ROC curve and AUC score for a given fold.

    Parameters
    ----------
    train_data : DataFrame
        Training data for the fold.
    validation_data : DataFrame
        Validation data for the fold.
    features : str | list[str]
        List of feature column names, or a single feature column name.
    target : str
        Target column name.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, float]
        FPR, TPR, AUC score.
    """
    X_train, y_train = train_data[features], train_data[target]
    X_validation, y_validation = validation_data[features], validation_data[target]

    model = LogisticRegressionCV(cv=2, max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    y_scores = model.predict_proba(X_validation)[:, 1]

    fpr, tpr, _ = roc_curve(y_validation, y_scores)
    roc_auc = auc(fpr, tpr)

    return fpr, tpr, roc_auc


def calculate_mean_roc_auc(
    roc_curves: list[tuple[np.ndarray, np.ndarray, float]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate the mean ROC curve and standard error.

    Parameters
    ----------
    roc_curves : List of tuples
        ROC curves (FPR, TPR, AUC) for each fold.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        Mean FPR, Mean TPR, Standard Error.
    """
    all_fpr = np.linspace(0, 1, 100)
    mean_tpr = np.zeros_like(all_fpr)

    tprs = []
    for fpr, tpr, _ in roc_curves:
        tprs.append(np.interp(all_fpr, fpr, tpr))
    mean_tpr = np.mean(tprs, axis=0)
    std_error = np.std(tprs, axis=0) / np.sqrt(len(tprs))

    return all_fpr, mean_tpr, std_error
