"""Generate univariate analysis plots and statistics for the Predictables app."""

import streamlit as st
import pandas as pd
import numpy as np
from predictables.app import (
    update_state,
    initialize_state,
    is_data_loaded,
    two_column_layout_with_spacers,
    histogram,
    boxplot,
    scatter,
    # build_models,
)
from predictables.app.plots.roc import roc_curve
from predictables.app.plots.quintile_lift import quintile_lift
from predictables.util import get_column_dtype, fmt_col_name
from bokeh.plotting import figure

st.set_page_config(
    page_title="PredicTables - Univariate",
    page_icon=None,
    layout="wide",  # or {"centered", "wide"}
    initial_sidebar_state="auto",  # or {"auto", "collapsed", "expanded"}
    menu_items={
        "Get help": None,  # or a url for this link to point to
        "Report a bug": None,  # or a url for this link to point to
        "About": """# PredicTables

        PredicTables is an app that provides a simple interface for exploring and analyzing data.
        """,
    },
)


def get_feat() -> str:
    """Return the feature variable."""
    return (
        f" for feature `{fmt_col_name(univariate_feature_variable)}`"
        if univariate_feature_variable != ""
        else " "
    )


def very_basic_analysis(
    X: pd.Series, y: pd.Series, feature_type: str, target_type: str
) -> None:
    """Generate very basic analysis for the univariate analysis."""
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
            p = scatter(
                X,
                y,
                fmt_col_name(univariate_feature_variable),
                fmt_col_name(target_variable),
            )

            st.bokeh_chart(p)

        elif target_type in ["categorical", "binary"]:
            p = boxplot(
                X,
                y,
                fmt_col_name(univariate_feature_variable),
                fmt_col_name(target_variable),
            )

            st.bokeh_chart(p)

def density_plot(X: pd.Series, y: pd.Series) -> None:
    """Generate kernel density plot by level of the target variable."""
    from scipy.stats import gaussian_kde

    def kde(x: pd.Series, N: int) -> tuple:
        """Calculate kernel density estimate."""
        kde_ = gaussian_kde(x)
        x = np.linspace(x.min(), x.max(), N)
        y = kde_(x)
        return x, y

    p = figure(
        title="Density Plot by Target Variable",
        x_axis_label=f"{fmt_col_name(univariate_feature_variable)}",
        y_axis_label="Density",
        width=750,
        height=450,
    )

    lines = []
    for i, level in enumerate(y.unique()):
        x = X[y == level]
        x0, y0 = kde(x, 100)
        lines.append(
            p.line(
                x0,
                y0,
                legend_label=f"{fmt_col_name(target_variable)} = {level}",
                line_width=3,
                line_color="skyblue" if i % 2 == 0 else "lightcoral",
            )
        )

    for i, line in enumerate(lines):
        # fill the area under the line
        p.patch(
            np.append(line.data_source.data["x"], line.data_source.data["x"][::-1]),
            np.append(
                line.data_source.data["y"],
                np.zeros_like(line.data_source.data["y"])[::-1],
            ),
            color="skyblue" if i % 2 == 0 else "lightcoral",
            fill_alpha=0.2,
        )

    p.legend.location = "top_right"
    p.legend.click_policy = "hide"

    st.bokeh_chart(p)




def stacked_bar_chart(X: pd.Series, y: pd.Series) -> None:
    """Generate a stacked bar chart showing the mean target variable for each level of the feature."""
    # Calculate the mean target variable for each level of the feature
    df = pd.DataFrame({univariate_feature_variable: X, target_variable: y})
    df = df.groupby(univariate_feature_variable)[target_variable].mean().reset_index()

    # Generate the stacked bar chart
    p = figure(
        title="Mean Target Variable by Feature Level",
        x_axis_label=f"{fmt_col_name(univariate_feature_variable)}",
        y_axis_label=f"Mean {fmt_col_name(target_variable)}",
        width=750,
        height=450,
    )

    p.vbar(
        x=univariate_feature_variable,
        top=target_variable,
        source=df,
        width=0.5,
        fill_alpha=0.7,
        line_color=None,
        fill_color="skyblue",
    )

    st.bokeh_chart(p)



def when_data_loaded() -> None:
    """Populate this page only when data are loaded."""
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

    st.markdown(
        f"## Univariate analysis {get_feat()} with target `{fmt_col_name(target_variable)}`"
    )

    # Extract X and y from the data
    X = st.session_state["data"][univariate_feature_variable]
    y = st.session_state["data"][target_variable]
    fold = st.session_state["data"]["fold"]

    df = pd.DataFrame(
        {univariate_feature_variable: X, target_variable: y, "fold": fold}
    )

    # == Meat of the univariate analysis starts here =====================
    feature_type = get_column_dtype(X)
    target_type = get_column_dtype(y)

    very_basic_analysis(X, y, feature_type, target_type)

    # == Univariate Analysis Graphs =====================
    if feature_type == "continuous" and target_type in ["categorical", "binary"]:
        st.markdown("### Density Plot")
        col1, col2 = two_column_layout_with_spacers()
        with col1:
            density_plot(X, y)

    elif feature_type in ["categorical", "binary"] and target_type in [
        "categorical",
        "binary",
    ]:
        st.markdown("### Stacked Bar Chart")
        col1, col2 = two_column_layout_with_spacers()
        with col1:
            stacked_bar_chart(X, y)

    # == ROC-AUC Curve =====================
    st.markdown("### ROC AUC Plot")
    p, roc_df = roc_curve(df, univariate_feature_variable, target_variable)
    col1, col2 = two_column_layout_with_spacers()

    col1.bokeh_chart(p)
    col2.dataframe(roc_df)

    # == Quintile Lift Plot =====================
    st.markdown("### Quintile Lift Plot")
    p, quintile_df = quintile_lift(X, y, fold)


    st.bokeh_chart(p)
    st.dataframe(quintile_df)


# Initialize state variables if needed
initialize_state()

target_variable = st.session_state["target_variable"]
univariate_feature_variable = st.session_state["univariate_feature_variable"]


if not is_data_loaded():
    st.markdown(
        "**Warning: ** No data is loaded. Return to the `Load Data` page before coming back."
    )
else:
    when_data_loaded()