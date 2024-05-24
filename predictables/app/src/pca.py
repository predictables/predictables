from sklearn.decomposition import PCA
import pandas as pd
import polars as pl
import polars.selectors as cs
import streamlit as st


@st.cache_resource
def pca(X: pd.DataFrame, n_components: int = 20) -> PCA:
    """Generate a PCA model."""
    # Preprocess the data to ensure numerical values have mean 0 and variance 1
    Xstd = pl.from_pandas(X).lazy()
    Xstd = (
        Xstd.select(
            [
                pl.when(pl.col(c).std() == 0)
                .then(pl.col(c))
                .otherwise((pl.col(c) - pl.col(c).mean()) / pl.col(c).std())
                .alias(c)
                for c in Xstd.select(cs.numeric()).columns
            ]
        )
        .collect()
        .to_pandas()
    )

    p = PCA(n_components=n_components, random_state=42, svd_solver="auto")
    p.fit(Xstd)
    return p
