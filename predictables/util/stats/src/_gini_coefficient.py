def gini_coefficient(observed: list, modeled: list) -> float:
    """Calculate the Gini coefficient between two distributions.

    This function is a wrapper for sklearn.metrics.roc_auc_score.

    Parameters
    ----------
    observed : list
        List of observed values
    modeled : list
        List of modeled values

    Returns
    -------
    float
        Gini coefficient between the two distributions
    """
    from sklearn.metrics import roc_auc_score  # type: ignore[import-untyped]

    # Gini coefficient calculation using AUC
    return 2 * roc_auc_score(observed, modeled) - 1
