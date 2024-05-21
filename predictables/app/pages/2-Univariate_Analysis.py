"""Generate univariate analysis plots and statistics for the Predictables app."""

import streamlit as st
import pandas as pd
from predictables.app import update_state, initialize_state, is_data_loaded
from predictables.util import get_column_dtype, fmt_col_name

# Initialize state variables if needed
initialize_state()

st.markdown("# Univariate Analysis")

if not is_data_loaded():
    st.markdown(
        "**Warning: ** No data is loaded. Return to the `Load Data` page before coming back."
    )
else:
    target_variable = st.session_state["target_variable"]
    univariate_feature_variable = st.selectbox(
        "Feature Variable",
        st.session_state["columns"],
        key="univariate-feature-variable",
        placeholder="Feature variable...",
        index=st.session_state["columns"].index(
            st.session_state["univariate_feature_variable"]
        )
        if st.session_state["univariate_feature_variable"]
        in st.session_state["columns"]
        else 0,
        on_change=lambda: update_state(
            "univariate_feature_variable", univariate_feature_variable
        ),
    )
    # Extract X and y from the data
    X = st.session_state["data"][univariate_feature_variable]
    y = st.session_state["data"][target_variable]

    # == Meat of the univariate analysis starts here =====================
    st.markdown(
        f"### Univariate analysis for `{fmt_col_name(univariate_feature_variable)}` and `{fmt_col_name(target_variable)}`"
    )

    feature_type = get_column_dtype(X)
    target_type = get_column_dtype(y)

    col1, col2 = st.columns(2)

    col1.markdown(f"##### `{fmt_col_name(univariate_feature_variable)}`")

    if feature_type == "continuous":
        skewness = X.skew().round(4)
        skewness_label = (
            "Symmetric"
            if -0.5 < skewness < 0.5
            else "Moderately Skewed"
            if -1 < skewness < 1
            else "Highly Skewed"
        )
        mean_over_median = (X.mean() / X.median()).round(4)
        extra_stats = {
            "Skewness": skewness,
            "Is it skewed?": skewness_label,
            "Mean/Median Ratio": mean_over_median,
        }
        col1.dataframe(
            pd.concat([X.describe().round(4), pd.Series(extra_stats)]).reset_index(),
            hide_index=True,
        )
    elif feature_type in ["categorical", "binary"]:
        col1.dataframe(X.value_counts())

    col2.markdown(f"##### `{fmt_col_name(target_variable)}`")

    if target_type == "continuous":
        col2.dataframe(y.describe())
    elif target_type in ["categorical", "binary"]:
        col2.dataframe(y.value_counts())