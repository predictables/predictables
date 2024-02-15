from typing import List, Optional, Tuple, Union

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import pandas as pd
import polars as pl
from sklearn.decomposition import PCA as sklearn_PCA  # type: ignore

from predictables.pca.src import (
    create_biplot,
    create_loading_plot,
    create_scree_plot,
    feature_importance,
    perform_pca,
    select_n_components_for_variance,
)
from predictables.util import to_pd_df


class PCA:
    """
    Principal Components Analysis class.

    Parameters
    ----------
    n_components : int, optional
        Number of components to retain, by default 10.
    df : Union[pd.DataFrame, pl.DataFrame, pl.LazyFrame], optional
        Dataset to fit PCA to, by default None. If None, pca.fit() must be called
        before any other methods.
    preprocess_data : bool, optional
        Whether to preprocess the data before fitting the PCA model, by default True.
    random_state : int, optional
        Random state for reproducibility, by default 42.
    pca : sklearn.decomposition.PCA, optional
        A fitted PCA object, by default None. If None, pca.fit() must be called
        before any other methods.
    features : list, optional
        A list of feature names, by default None. If None, the feature names are
        set to the column names of df.
    plotting_backend : str, optional
        The plotting backend to use, by default "matplotlib". "plotly" integration is
        planned, but not yet implemented.
    **kwargs
        Keyword arguments to pass to sklearn.decomposition.PCA.

    Attributes
    ----------
    n_components : int
        Number of components to retain.
    df : Union[pd.DataFrame, pl.DataFrame, pl.LazyFrame]
        Dataset to fit PCA to.
    preprocess_data : bool
        Whether to preprocess the data before fitting the PCA model.
    random_state : int
        Random state for reproducibility.
    pca : sklearn.decomposition.PCA
        A fitted PCA object.
    features : list
        A list of feature names.
    plotting_backend : str
        The plotting backend to use.
    explained_variance : list
        The explained variance for each component.

    Methods
    -------
    set_n_components
        Sets the number of components to retain. Can either pass `n_components` or
        `variance_threshold`.
    _refitPCA
        Refits the PCA model. Private method.
    fit_pca
        Fits a PCA model to the provided dataset.
    transform_pca
        Transforms the provided dataset using the fitted PCA model.
    get_principal_components
        Returns the principal components.
    scree
        Creates a scree plot to help determine the number of principal components to
        retain.
    biplot
        Creates a biplot to visualize the features in the first two principal
        components.
    loading_plot
        Original plot from Andy's noggin. Stacks the absolute value loadings for each
        feature across the first `n_components` principal components. The features are
        sorted by the magnitude of their average loading across the first `n_components`
        principal components. The size of the bars in total is a measure of the feature
        importance (for explaining the variance in the dataset).
    feature_importance
        Returns a sorted table of features sorted by the magnitude of their loading
        vectors with `self.n_components` components after being scaled by the explained
        variance of each component.

    Examples
    --------
    >>> from predictables.pca import PCA
    >>> from sklearn.datasets import load_breast_cancer

    >>> # Load the breast cancer dataset from sklearn
    >>> X = load_breast_cancer()['data']
    >>> X.shape
    (569, 30)

    >>> # Create a PCA object
    >>> pca = PCA(X, n_components=10)
    >>> # Call the object to see the number of components:
    >>> pca
    PCA[10 components]

    >>> # Set a more reasonable number of components to retain
    >>> pca.set_n_components(3)
    >>> pca
    PCA[3 components]

    >>> # Change your mind, decide to reset the number of components to retain 95% of
    >>> # the variance in the data
    >>> pca.set_n_components(variance_threshold=0.95)
    >>> pca
    PCA[10 components]

    >>> # Confirm that the explained variance is 95% by printing the explained variance
    >>> pca.explained_variance
    """

    def __init__(
        self,
        n_components: int = 10,
        df: Optional[Union[pd.DataFrame, pl.DataFrame, pl.LazyFrame]] = None,
        preprocess_data: bool = True,
        random_state: int = 42,
        pca: Optional[sklearn_PCA] = None,
        features: Optional[List[str]] = None,  # type: ignore
        plotting_backend: str = "matplotlib",  # TODO: Add plotly support
        **kwargs,
    ):
        # Set attributes
        self.n_components = n_components
        self.df = to_pd_df(df) if df is not None else pd.DataFrame()

        self.preprocess_data = preprocess_data
        self.random_state = random_state
        self.plotting_backend = plotting_backend

        # Fit PCA if df is provided and pca is not provided
        if df is not None and pca is None:
            self.pca = self.fit_pca(df=df, return_pca_obj=True, **kwargs)
        else:
            self.pca = pca

        # Set explained variance
        if self.pca is not None:
            self.explained_variance = self.pca.explained_variance_ratio_.tolist()
        else:
            self.explained_variance = None

        # Set features
        if self.df is not None:
            self.features = self.df.columns.tolist()
        elif features is not None:
            self.features = features
        else:
            self.features = None

    def __repr__(self) -> str:
        return f"PCA[{self.n_components} components]"

    def set_n_components(
        self,
        n_components: Optional[int] = None,
        variance_threshold: Optional[float] = None,
    ):
        """
        Sets the number of components to retain. Can either pass `n_components` or
        `variance_threshold`.

        Parameters
        ----------
        n_components : int, optional
            Number of components to retain, by default None.
        variance_threshold : float, optional
            Threshold for the cumulative variance to be retained, by default None.

        Raises
        ------
        ValueError
            If both `n_components` and `variance_threshold` are None.
        """
        # Set n_components
        if n_components is not None:
            self.n_components = n_components
            self._refitPCA()

        # Set n_components using variance_threshold
        elif variance_threshold is not None:
            self.n_components = select_n_components_for_variance(
                X=self.df, variance_threshold=variance_threshold
            )
            self._refitPCA()

        # If both are None, do nothing and print a message
        else:
            raise ValueError(
                "Either n_components or variance_threshold must be provided"
            )

    def _refitPCA(self):
        if self.pca is not None:
            self.pca = self.fit_pca(df=self.df, return_pca_obj=True)
            self.explained_variance = self.pca.explained_variance_ratio_.tolist()
            self.features = self.df.columns.tolist()

    def fit_pca(
        self,
        df: Optional[Union[pd.DataFrame, pl.DataFrame, pl.LazyFrame]] = None,
        return_pca_obj: bool = False,
    ):
        """
        Fits a PCA model to the provided dataset.

        Parameters
        ----------
        df : Union[pd.DataFrame, pl.DataFrame, pl.LazyFrame]
            Dataset of shape (n_samples, n_features) to fit PCA to.

        Returns
        -------
        sklearn.decomposition.PCA
            The fitted PCA object.
        """
        # Set df
        if df is None:
            df = self.df

        # Fit PCA
        return perform_pca(
            X_train=df,
            n_components=self.n_components,
            return_pca_obj=return_pca_obj,
            preprocess_data=self.preprocess_data,
            random_state=self.random_state,
        )

    def transform_pca(
        self,
        df: Optional[Union[pd.DataFrame, pl.DataFrame, pl.LazyFrame]] = None,
    ):
        """
        Transforms the provided dataset using the fitted PCA model.

        Parameters
        ----------
        df : Union[pd.DataFrame, pl.DataFrame, pl.LazyFrame]
            Dataset of shape (n_samples, n_features) to transform.

        Returns
        -------
        pd.DataFrame
            The transformed dataset.
        """
        # Set df
        if df is None:
            df = self.df

        # Transform dataset
        return self.pca.transform(df)

    def get_principal_components(
        self, components: Optional[Union[List[int], int]] = None
    ):
        """
        Returns the principal components.

        Parameters
        ----------
        components : Union[List[int], int, None], optional
            The components to return. If None, returns all components, by default None.

        Returns
        -------
        pd.DataFrame
            The principal components. The index is the component number, and the
            columns are the features.
        """
        # Fit PCA if not already fitted
        if self.pca is None:
            self.pca = self.fit_pca(df=self.df, return_pca_obj=True)

        # Get principal components
        principal_components = pd.DataFrame(self.pca.components_, columns=self.features)

        # Return all components if components is None
        if components is None:
            return principal_components

        # Return the specified components otherwise
        else:
            return principal_components.iloc[components]

    def scree(
        self,
        df: Optional[Union[pd.DataFrame, pl.DataFrame, pl.LazyFrame]] = None,
        variance_levels: Optional[List[float]] = None,
        y_pos_adjustment: float = 0.1,
        ax: Optional[matplotlib.axes.Axes] = None,
        figsize: Tuple[int, int] = (10, 7),
    ) -> matplotlib.axes.Axes:
        """
        Creates a scree plot.

        Parameters
        ----------
        df : Union[pd.DataFrame, pl.DataFrame, pl.LazyFrame], optional
            Dataset of shape (n_samples, n_features) to create the scree plot for, by
            default None. If None, uses the dataset provided when instantiating the
            PCA object.
        variance_levels : list of float, optional
            A list of levels at which to annotate the cumulative variance.
            For example, [0.75, 0.90, 0.95, 0.99]. By default, it's set to
            [0.75, 0.90, 0.95, 0.99].
        y_pos_adjustment : float, optional
            Adjustment for the y position of the annotations, by default 0.05.
        ax : matplotlib.axes.Axes, optional
            The Axes object to plot on. If None, a new figure and Axes object is
            created.
        figsize : tuple of int, optional
            The figure size, by default (10, 7).


        Returns
        -------
        matplotlib.axes.Axes
            The Axes object with the scree plot.
        """
        if df is None:
            if self.df is None:
                raise ValueError("df must be provided")
            else:
                df = self.df

        if variance_levels is None:
            variance_levels = [0.75, 0.90, 0.95, 0.99]

        # Create new axes if ax is None
        if ax is None:
            _, ax = plt.subplots(figsize=figsize)

        # Create scree plot
        return create_scree_plot(
            X=(
                df.collect().to_numpy()
                if isinstance(df, pl.LazyFrame)
                else df.to_numpy()
            ),
            variance_levels=variance_levels,
            y_pos_adjustment=y_pos_adjustment,
            ax=ax,
            figsize=figsize,
        )

    def biplot(
        self,
        loading_threshold: float = 0.2,
        ax: Optional[Axes] = None,
        figsize: Tuple[int, int] = (10, 10),
        use_limits: bool = True,
    ):
        """
        Creates a biplot.

        Returns
        -------
        matplotlib.axes.Axes
            The Axes object with the biplot.
        """
        backend = self.plotting_backend

        # Fit PCA if not already fitted
        if self.pca is None:
            self.pca = self.fit_pca(df=self.df, return_pca_obj=True)

        # Create biplot
        ax = create_biplot(
            pca=self.pca,
            feature_names=self.features,
            ax=ax,
            loading_threshold=loading_threshold,
            figsize=figsize,
            backend=backend,
            use_limits=use_limits,
        )

        return ax

    def loading_plot(
        self,
        n_components=10,
        max_features=50,
        average_loading_threshold=0.01,
        ax=None,
        fig=None,
        figsize=(10, 7),
        bar_alpha=0.8,
        bar_width=0.9,
        main_title_fontsize=13,
        main_title_fontweight="bold",
        sub_title_fontsize=10,
        legend_fontsize=9,
        x_ticks_rotation=45,
        x_label_fontsize=10,
        y_label_fontsize=10,
        include_legend=True,
        drop_legend_when_n_features=15,
        return_ax=False,
    ):
        backend = self.plotting_backend

        # Fit PCA if not already fitted
        if self.pca is None:
            self.pca = self.fit_pca(df=self.df, return_pca_obj=True)

        if ax is None:
            _, ax = plt.subplots(figsize=figsize)

        # Create loading plot
        ax = create_loading_plot(
            pca=self.pca,
            feature_names=self.features,
            n_components=(
                n_components if n_components is not None else self.n_components
            ),
            ax=ax,
            fig=fig,
            max_features=max_features,
            average_loading_threshold=average_loading_threshold,
            figsize=figsize,
            backend=backend,
            bar_alpha=bar_alpha,
            bar_width=bar_width,
            main_title_fontsize=main_title_fontsize,
            main_title_fontweight=main_title_fontweight,
            sub_title_fontsize=sub_title_fontsize,
            legend_fontsize=legend_fontsize,
            x_ticks_rotation=x_ticks_rotation,
            x_label_fontsize=x_label_fontsize,
            y_label_fontsize=y_label_fontsize,
            include_legend=include_legend,
            drop_legend_when_n_features=drop_legend_when_n_features,
        )

        if return_ax:
            return ax
        else:
            plt.show()

    def feature_importance(self) -> pd.DataFrame:
        """
        Returns a sorted table of features sorted by the magnitude of
        their loading vectors with self.n_components components after
        being scaled by the explained variance of each component.
        """
        if self.pca is None:
            self.pca = self.fit_pca(df=self.df, return_pca_obj=True)

        # Get feature importance
        fi = pd.DataFrame(
            feature_importance(self.pca),
            index=self.features,
            columns=["Feature Importance"],
        )

        # Sort by feature importance
        fi = fi.sort_values(by="Feature Importance", ascending=False)

        return fi
