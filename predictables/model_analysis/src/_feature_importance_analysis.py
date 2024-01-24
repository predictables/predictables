import numpy as np
import pandas as pd
import shap
import sklearn


def feature_importance_analysis(
    model: sklearn.base.BaseEstimator, X: pd.DataFrame
) -> pd.DataFrame:
    """
    Analyzes and returns feature importance scores using SHAP values.

    Parameters
    ----------
    model : sklearn.base.BaseEstimator
        A trained model object that inherits from sklearn.base.BaseEstimator.
    X : pd.DataFrame
        A Pandas DataFrame containing the features used to train the model.

    Returns
    -------
    feature_importance_df : pd.DataFrame
        A Pandas DataFrame containing the feature names and their SHAP values.
    """
    # Compute SHAP values
    explainer = shap.Explainer(model, X)
    shap_values = explainer.shap_values(X)

    # Calculate mean absolute SHAP values for each feature
    shap_sum = np.abs(shap_values).mean(axis=0)
    feature_importance_df = pd.DataFrame(
        {"feature": X.columns, "shap_importance": shap_sum}
    ).sort_values(by="shap_importance", ascending=False)

    return feature_importance_df
