from typing import Union, Optional

import pandas as pd
import polars as pl
from sklearn.decomposition import PCA  # type: ignore

from predictables.util import to_pd_df

from ._preprocessing import preprocess_data_for_pca


def perform_pca(
    X_train: Union[pd.DataFrame, pl.DataFrame, pl.LazyFrame],
    X_val: Optional[Union[pd.DataFrame, pl.DataFrame, pl.LazyFrame]] = None,
    X_test: Optional[Union[pd.DataFrame, pl.DataFrame, pl.LazyFrame]] = None,
    n_components: int = 10,
    return_pca_obj: bool = False,
    preprocess_data: bool = True,
    random_state: int = 42,
    **kwargs,
) -> tuple:
    """
    Performs Principal Component Analysis (PCA) on the provided datasets.

    This function fits a PCA model to the training data and applies the transformation
    to the training, validation, and test datasets. The number of principal components
    is determined by the `n_components` parameter.

    Parameters
    ----------
    X_train : Union[pd.DataFrame, pl.DataFrame, pl.LazyFrame]
        Training dataset of shape (n_samples, n_features).
    X_val : Union[pd.DataFrame, pl.DataFrame, pl.LazyFrame], optional
        Validation dataset of shape (n_samples_val, n_features). If not
        provided, the function will not transform the validation dataset,
        by default None.
    X_test : Union[pd.DataFrame, pl.DataFrame, pl.LazyFrame], optional
        Test dataset of shape (n_samples_test, n_features). If not provided, the
        function will not transform the test dataset, by default None.
    n_components : int, optional
        Number of principal components to keep. Defaults to 10.
    return_pca_obj : bool, optional
        Whether to return the PCA object, by default False.
    preprocess_data : bool, optional
        Whether to preprocess the data before applying PCA, by default True.
    random_state : int, optional
        Random state for reproducibility, by default 42.
    kwargs : optional
        Additional keyword arguments to pass to the PCA object.

    Returns
    -------
    X_train_pca : pl.LazyFrame, optional
        Transformed training dataset. Only returned if `return_pca_obj` is False.
    X_val_pca : pl.LazyFrame, optional
        Transformed validation dataset. Only returned if `return_pca_obj` is False,
        and `X_val` is not None.
    X_test_pca : pl.LazyFrame, optional
        Transformed test dataset. Only returned if `return_pca_obj` is False, and
        `X_test` is not None.
    pca : PCA object, optional
        The fitted PCA object from scikit-learn. Only returned if `return_pca_obj`
        is True.

    Notes
    -----
    The function does not handle missing values or categorical features. Ensure that
    the datasets are preprocessed accordingly before applying PCA.

    The PCA transformation is irreversible; information is lost after applying PCA.

    While you may pass train, validation, and test datasets to this function, it is
    not necessary. The fitted object can be used to transform new data. It is, however,
    recommended to split the data into training, validation, and test sets before
    applying PCA.

    Data should be passed to this function as either a pandas DataFrame or a polars
    DataFrame or LazyFrame. Do not pass a numpy array. I hate numpy arrays.

    Examples
    --------
    >>> from sklearn.datasets import load_iris
    >>> X, _ = load_iris(return_X_y=True)
    >>> X_train, X_test = X[:100], X[100:]
    >>> X_train_pca, _, X_test_pca = perform_pca(
            X_train, X_test, X_test, n_components=2
        )

    Raises
    ------
    ValueError
        If `n_components` is greater than the number of features in the dataset.

    See Also
    --------
    sklearn.decomposition.PCA : PCA implementation used in this function.
    """
    # Convert to pandas dataframe
    X_train = to_pd_df(X_train)
    if X_val is not None:
        X_val = to_pd_df(X_val)
    if X_test is not None:
        X_test = to_pd_df(X_test)

    if n_components > X_train.shape[1]:
        raise ValueError("n_components cannot be greater than the number of features")

    pca = PCA(n_components=n_components, random_state=random_state, **kwargs)

    # Preprocess the data
    if preprocess_data:
        X_train = preprocess_data_for_pca(X_train)
        if X_val is not None:
            X_val = preprocess_data_for_pca(X_val)
        if X_test is not None:
            X_test = preprocess_data_for_pca(X_test)

    if return_pca_obj:
        return pca.fit(to_pd_df(X_train))

    # Fit and transform X_train, X_val, and X_test
    X_train_pca = pca.fit_transform(to_pd_df(X_train))
    X_val_pca = pca.transform(to_pd_df(X_val)) if X_val is not None else None
    X_test_pca = pca.transform(to_pd_df(X_test)) if X_test is not None else None

    # If X_test was provided, transform it and add it to the output,
    # regardless of whether X_val was provided
    out = X_train_pca if X_val_pca is None else (X_train_pca, X_val_pca)
    return (*out, X_test_pca) if X_test_pca is not None else out
