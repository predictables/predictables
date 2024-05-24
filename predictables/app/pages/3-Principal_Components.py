import streamlit as st

import polars as pl
import polars.selectors as cs
import pandas as pd
from predictables.util import get_column_dtype

from predictables.app.src.util import get_data
from predictables.app.src.pca import pca as _pca
from predictables.app.plots.pca_loading.src.loading_plot import loading_plot
from predictables.app.plots.scree_plot.src.scree_plot import scree_plot
from predictables.app.plots.biplot.biplot import biplot

target_variable = st.session_state["target_variable"]

X = pl.from_pandas(get_data()).lazy().drop(["fold", target_variable])
X = X.drop(X.select(cs.temporal()).columns)

for c in X.columns:
    if get_column_dtype(X.collect()[c]) in ["continuous", "binary", "integer"]:
        X = X.with_columns(pl.col(c).cast(pl.Float32))
    elif get_column_dtype(X.collect()[c]) in ["categorical", "string"]:
        X = X.with_columns(pl.col(c).cast(pl.Categorical))

y = get_data()[target_variable]
fold = get_data()["fold"]
pca = _pca(X.collect().to_pandas(), 20)


st.markdown("# Principal Components")

st.markdown("## Loading plot")
p = loading_plot(
    pca, n_components=pca.n_components_, feature_names=X.columns, max_features=50
)
st.bokeh_chart(p)


st.markdown("## Scree plot")
p = scree_plot(X.collect().to_pandas(), variance_levels=[0.8, 0.9, 0.95, 0.99])
st.bokeh_chart(p)

st.markdown("## Biplot")
p = biplot(X.collect().to_pandas())
st.bokeh_chart(p)