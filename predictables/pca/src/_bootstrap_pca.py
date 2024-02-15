from typing import Union, Dict

import pandas as pd
import polars as pl
from sklearn.utils import resample  # type: ignore

from predictables.util import to_pl_df

from ._perform_pca import perform_pca


def bootstrap_pca(
    data: Union[pd.DataFrame, pl.DataFrame, pl.LazyFrame],
    n_components: int,
    n_bootstraps: int = 1000,
    random_state: int = 42,
) -> dict:
    """
    Performs PCA on multiple bootstrapped samples of the dataset.

    Parameters
    ----------
    data : Union[pd.DataFrame, pl.DataFrame, pl.LazyFrame]
        Original dataset for PCA.
    n_components : int
        Number of principal components to compute.
    n_bootstraps : int, optional
        Number of bootstrapped samples to generate. Defaults to 1000.
    random_state : int, optional
        Random state for reproducibility. Defaults to 42.

    Returns
    -------
    bootstrapped_results : dict
        Dictionary containing loadings and explained variance for each bootstrapped
        sample.

    Notes
    -----
    1. The function uses scikit-learn's `resample` function to generate bootstrapped
       samples.
    2. The function uses `PredicTable.PCA.src.perform_pca` to perform PCA on each
       bootstrapped sample.

    Examples
    --------
    >>> from sklearn.datasets import load_iris
    >>> X, _ = load_iris(return_X_y=True)
    >>> bootstrapped_results = bootstrap_pca(X, n_components=2)
    >>> bootstrapped_results["loadings"][0]
    array([[ 0.36138659, -0.08452251,  0.85667061,  0.3582892 ],
           [ 0.65658877,  0.73016143, -0.17337266, -0.07548102]])
    >>> bootstrapped_results["explained_variance"][0]
    array([0.92461872, 0.05306648])

    Raises
    ------
    ValueError
        If `data` is not a 2-dimensional array.
    ValueError
        If `n_components` is greater than the number of features in the dataset.

    See Also
    --------
    sklearn.utils.resample : Function used to generate bootstrapped samples.
    PredicTable.PCA.src.perform_pca : Function used to perform PCA on each bootstrapped
    sample.
    """
    # Convert `data` to polars DataFrame if it isn't already:
    data = to_pl_df(data)

    # Test that inputs are valid
    if len(data.shape) != 2:
        raise ValueError("Data must be a 2-dimensional array.")

    if n_components > data.shape[1]:
        raise ValueError("n_components cannot be greater than the number of features.")

    # Initialize dictionary to store results
    bootstrapped_results: Dict[str, list] = {"loadings": [], "explained_variance": []}

    # Generate bootstrapped samples
    for _ in range(n_bootstraps):
        sample = resample(data, replace=True, random_state=random_state)

        # `perform_pca` on each bootstrapped sample
        pca_obj = perform_pca(sample, n_components=n_components, return_pca_obj=True)

        if isinstance(pca_obj, tuple):
            pca_obj = pca_obj[0]

        bootstrapped_results["loadings"].append(pca_obj.components_)  # type: ignore
        bootstrapped_results["explained_variance"].append(
            pca_obj.explained_variance_ratio_  # type: ignore
        )

    return bootstrapped_results
