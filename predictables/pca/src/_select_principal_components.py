import numpy as np
from sklearn.decomposition import PCA


def select_n_components_for_variance(X, variance_threshold=0.95):
    """
    Selects the number of principal components to retain enough variance.

    This function calculates the number of principal components that are needed
    to retain a specified percentage of the variance in the dataset.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Training dataset.
    variance_threshold : float, optional
        Threshold for the cumulative variance to be retained, by default 0.95.

    Returns
    -------
    n_components : int
        The number of principal components to retain.

    Notes
    -----
    The function uses PCA to calculate the explained variance ratio for each component
    and selects the minimum number of components that satisfy the cumulative variance
    threshold.

    Examples
    --------
    >>> from sklearn.datasets import load_iris
    >>> X, _ = load_iris(return_X_y=True)
    >>> n_components = select_n_components_for_variance(X, variance_threshold=0.95)
    """
    pca = PCA().fit(X)
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    n_components = np.argmax(cumulative_variance >= variance_threshold) + 1
    return n_components
