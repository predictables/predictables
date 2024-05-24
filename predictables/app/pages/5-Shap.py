"""Define the Shap page of the app."""

import streamlit as st
from streamlit_shap import st_shap
import shap
import polars as pl
import numpy as np

from predictables.app import (
    fit_model,
    shap_values as _shap_values,
    shap_bar_plot,
    shap_beeswarm_plot,
)

from predictables.app.src.util import get_data

data = pl.from_pandas(get_data()).lazy()
max_fold = data.select(pl.col("fold").max()).collect().to_pandas().iloc[0, 0]
lf_train = data.filter(pl.col("fold") < max_fold)
lf_test = data.filter(pl.col("fold") == max_fold)

X_train = (
    lf_train.drop(["fold", st.session_state["target_variable"]]).collect().to_pandas()
)
y_train = lf_train.select(st.session_state["target_variable"]).collect().to_pandas()
X_test = (
    lf_test.drop(["fold", st.session_state["target_variable"]]).collect().to_pandas()
)
y_test = lf_test.select(st.session_state["target_variable"]).collect().to_pandas()

model = fit_model(X_train, y_train, X_test, y_test)
shap_values = _shap_values(model, X_train, X_test)

st.markdown("# Shap Analysis")

initial_number_of_features = max(int(np.sqrt(X_train.shape[1])), 10)


col1, col2 = st.columns(2)

with col1:
    st.markdown("### Global Bar Plot")

    col1a, col1b, col1c = st.columns(3)
    with col1a:
        n_features_in_bar_plot = st.number_input(
            "Number of features in bar plot",
            min_value=1,
            max_value=X_train.shape[1],
            value=initial_number_of_features,
            step=1,
        )

    with col1b:
        n_cohorts_to_estimate = st.number_input(
            "Number of cohorts to estimate", min_value=1, max_value=10, value=1, step=1
        )

    with col1c:
        use_feature_clustering = st.checkbox("Use feature clustering", value=False)
        clustering_cutoff = st.number_input(
            "Clustering cutoff", min_value=0.0, max_value=1.0, value=0.5, step=0.01
        )

    shap_bar_plot(
        shap_values,
        cohorts=1 if use_feature_clustering else n_cohorts_to_estimate,
        max_display=n_features_in_bar_plot,
        use_clustering=use_feature_clustering,
        clustering_cutoff=clustering_cutoff,
        X=X_train,
        y=y_train,
    )

with col2:
    st.markdown("### Beeswarm Plot")

    col2a, col2b, col2c = st.columns(3)

    with col2a:
        n_features_in_beeswarm = st.number_input(
            "Number of features in beeswarm plot",
            min_value=2,
            max_value=X_train.shape[1],
            value=initial_number_of_features,
            step=1,
        )

    with col2b:
        order_by = st.selectbox("Order by", ["mean", "sum", "max", "min"], index=0)

    reverse_order = col2c.checkbox("Reverse order", value=False)

    shap_beeswarm_plot(
        shap_values,
        max_display=n_features_in_beeswarm,
        order_by=order_by,
        reverse_order=reverse_order,
    )

st.markdown("### Summary Plot")
st_shap(shap.plots.waterfall(shap_values[0]), height=400)
st_shap(shap.plots.waterfall(shap_values[1]), height=400)

st.write("See how the model prediction was determined for each quote.")

quote_number = st.selectbox("Select the quote number", ["1", "2", "3", "4", "5"])

st.write(
    "Select a group of quotes by control variable to see how the Shapley values are distributed for each feature in that subset of the data."
)

naics_code = st.selectbox("Select the NAICS code", ["11", "22", "33", "44-45", "54"])