"""Build the quintile lift dataframe for the Predictables app."""

from __future__ import annotations
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


def quintile_lift_df(
    y_train: pd.Series,
    X_test: pd.Series | pd.DataFrame,
    y_test: pd.Series,
    model: RandomForestClassifier,
) -> pd.DataFrame:
    """Build the quintile lift dataframe for the Predictables app."""
    yhat_test = model.predict_proba(X_test.to_numpy().reshape(-1, 1))[:, 1]

    df = pd.DataFrame({"actual": y_test, "random_forest_model": yhat_test})
    df["naive_model"] = y_train.mean()

    # Create quintile buckets for y_test (actual data)
    quintile_bucket, bins = pd.qcut(
        yhat_test, 5, labels=False, retbins=True, duplicates="drop"
    )

    df["quintile"] = quintile_bucket

    # Calculate the mean of the actual values, naive model and model predictions for each quintile
    df = (
        df.groupby("quintile")
        .agg({"actual": "mean", "naive_model": "mean", "random_forest_model": "mean"})
        .reset_index()
    )

    df["model_pct_error"] = (df["random_forest_model"] - df["actual"]) / df["actual"]
    df["naive_pct_error"] = (df["naive_model"] - df["actual"]) / df["actual"]

    return df
