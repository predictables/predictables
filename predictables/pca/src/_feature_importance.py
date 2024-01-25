import numpy as np


def pca_feature_importance(pca, normalize: bool = True):
    """
    Calculates feature importance scores based on PCA.

    Parameters
    ----------
    pca : PCA object
        The fitted PCA object from scikit-learn.
    normalize : bool, optional
        Whether to normalize the feature importance scores, by default True.

    Returns
    -------
    np.ndarray
        The feature importance scores.

    Notes
    -----
    The feature importance scores are calculated as the sum of squared loadings
    across all components for each feature.
    """
    # Calculate squared loadings (feature contributions)
    loadings = pca.components_**2

    # Sum across all components
    feature_importances = np.sum(loadings, axis=0)

    # Normalize
    if normalize:
        feature_importances = feature_importances / np.sum(feature_importances)

    return feature_importances
