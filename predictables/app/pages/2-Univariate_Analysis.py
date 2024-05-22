"""Generate univariate analysis plots and statistics for the Predictables app."""

import streamlit as st
import pandas as pd
from predictables.app import (
    update_state,
    initialize_state,
    is_data_loaded,
    two_column_layout_with_spacers,
    histogram,
    boxplot,
    scatter,
)
from predictables.util import get_column_dtype, fmt_col_name
from predictables.univariate import Univariate


# Initialize state variables if needed
initialize_state()

target_variable = st.session_state["target_variable"]
univariate_feature_variable = st.session_state["univariate_feature_variable"]
feat = (
    f" for feature `{fmt_col_name(univariate_feature_variable)}`"
    if univariate_feature_variable != ""
    else " "
)
st.markdown(
    f"## Univariate analysis {feat} with target `{fmt_col_name(target_variable)}`"
)

if not is_data_loaded():
    st.markdown(
        "**Warning: ** No data is loaded. Return to the `Load Data` page before coming back."
    )
else:
    idx = (
        st.session_state["columns"].index(
            st.session_state["univariate_feature_variable"]
        )
        if st.session_state["univariate_feature_variable"]
        in st.session_state["columns"]
        else 0
    )

    update_state("univariate_feature_variable", st.session_state["columns"][idx])

    univariate_feature_variable = st.selectbox(
        "Feature Variable",
        st.session_state["columns"],
        key="univariate-feature-variable",
        placeholder="Feature variable...",
        index=idx,
        on_change=lambda: update_state(
            "univariate_feature_variable", univariate_feature_variable
        ),
    )
    # Extract X and y from the data
    X = st.session_state["data"][univariate_feature_variable]
    y = st.session_state["data"][target_variable]

    # == Meat of the univariate analysis starts here =====================
    feature_type = get_column_dtype(X)
    target_type = get_column_dtype(y)

    col1, col2 = two_column_layout_with_spacers()

    with col1:
        st.markdown(
            f"##### `{fmt_col_name(univariate_feature_variable)}` - {feature_type}"
        )

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
                "skew": skewness,
                "is_skew": skewness_label,
                "mean/median": mean_over_median,
            }
            st.dataframe(
                pd.concat([X.describe().round(4), pd.Series(extra_stats)])
                .to_frame()
                .rename(columns={0: f"{fmt_col_name(univariate_feature_variable)}"})[
                    fmt_col_name(univariate_feature_variable)
                ],
                use_container_width=True,
            )

        elif feature_type in ["categorical", "binary"]:
            st.dataframe(X.value_counts(), use_container_width=True)

    with col2:
        st.markdown(f"##### `{fmt_col_name(target_variable)}` - {target_type}")

        if target_type == "continuous":
            st.dataframe(y.describe(), use_container_width=True)
        elif target_type in ["categorical", "binary"]:
            st.dataframe(y.value_counts(), use_container_width=True)

    col1, col2 = two_column_layout_with_spacers()
    with col1:
        if feature_type == "continuous":
            # # Calculate and plot a histogram of the continuous feature

            p = histogram(
                X,
                fmt_col_name(univariate_feature_variable),
                f"Distribution of `{fmt_col_name(univariate_feature_variable)}`",
            )

            st.bokeh_chart(p)

    with col2:
        if target_type == "continuous":
            p = scatter()
            st.bokeh_chart(p)

        elif target_type in ["categorical", "binary"]:
            p = boxplot(
                X,
                y.map({0: "No Hit", 1: "Hit"}),
                fmt_col_name(univariate_feature_variable),
                fmt_col_name(target_variable),
                f"Boxplot of `{fmt_col_name(univariate_feature_variable)}` vs `{fmt_col_name(target_variable)}`",
            )

            st.bokeh_chart(p)
