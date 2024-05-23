"""Fit a random forest model for the quintile lift plot."""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier


def rf_model(
    X_train: pd.Series,
    y_train: pd.Series,
    n_estimators: int = 100,
    random_state: int = 42,
) -> RandomForestClassifier:
    """Fit a random forest model."""
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    model.fit(X_train.to_numpy().reshape(-1, 1), y_train.to_numpy())

    return model