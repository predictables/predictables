"""Generate univariate analysis plots and statistics for the Predictables app."""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
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
from predictables.app.plots.roc import (
    prepare_roc_data,
    calculate_roc_auc,
    calculate_mean_roc_auc,
    generate_roc_auc_plot,
)
from predictables.util import get_column_dtype, fmt_col_name


from bokeh.plotting import figure


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

def density_plot(X: pd.Series, y: pd.Series, folds: pd.Series) -> None:
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

    # for fold in folds.unique():
    #     for level in y.unique():
    #         x0, y0 = kde(X[(y == level) & (folds == fold)], 100)
    #         p.line(
    #             x0,
    #             y0,
    #             line_width=1,
    #             line_color="grey",
    #             line_dash="dashed",
    #             legend_label=f"Fold {fold} - {fmt_col_name(target_variable)} = {level}",
    #         )

    p.legend.location = "top_right"
    p.legend.click_policy = "hide"

    st.bokeh_chart(p)


def roc_curve(
    df: pd.DataFrame, univariate_feature_variable: str, target_variable: str
) -> None:
    """Generate ROC curve for the univariate analysis."""
    col1, col2 = two_column_layout_with_spacers()
    with col1:
        ## ROC Curve
        roc_data = prepare_roc_data(df, use_time_series_validation=True)

        # Calculate ROC AUC for each fold
        roc_curves = []
        features = [univariate_feature_variable]
        target = target_variable

        for train_data, validation_data in roc_data:
            roc_curve = calculate_roc_auc(train_data, validation_data, features, target)
            roc_curves.append(roc_curve)

        # Calculate mean ROC AUC and standard error
        mean_fpr, mean_tpr, std_error = calculate_mean_roc_auc(roc_curves)

        # Generate ROC AUC plot
        p = generate_roc_auc_plot(roc_curves, mean_fpr, mean_tpr, std_error)
        st.bokeh_chart(p)

    with col2:
        st.markdown("### ROC AUC Statistics")
        df = pd.DataFrame(
            {
                "Fold": [f"Fold-{i}" for i in range(1, len(roc_curves) + 1)]
                + ["Mean"]
                + ["Std Error"],
                "ROC AUC": [roc_auc for _, _, roc_auc in roc_curves]
                + [np.mean([roc_auc for _, _, roc_auc in roc_curves])]
                + [
                    np.std([roc_auc for _, _, roc_auc in roc_curves])
                    / np.sqrt(len(roc_curves))
                ],
            }
        ).set_index("Fold")

        df["ROC AUC"] = df["ROC AUC"].apply(lambda x: f"{x:.1%}")

        st.dataframe(df)


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


def lift_plot(X: pd.Series, y: pd.Series, folds: pd.Series) -> None:
    """Generate a quintile lift plot for the univariate analysis."""
    p = figure(
        title="Quintile Lift Plot",
        x_axis_label="Quintile",
        y_axis_label="Lift",
        width=750,
        height=450,
    )

    for fold in folds.drop_duplicates().sort_values().tolist()[:-1]:
        x_train = X[folds <= fold]
        x_test = X[folds == fold + 1]

        y_train = y[folds <= fold]
        y_test = y[folds == fold + 1]

        naive_model = np.mean(y_train)

        quintile_df = pd.DataFrame(
            {
                "quintile": [f"Q{i}" for i in range(1, 6)],
                "mean_target": [y_test.quantile(i / 5) for i in range(1, 6)],
                "naive_model": [naive_model for _ in range(5)],
            }
        )

        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(x_train.to_numpy().reshape(-1, 1), y_train)
        yhat_test = model.predict_proba(x_test.to_numpy().reshape(-1, 1))[:, 1]

        # Create quintile buckets and adjust labels accordingly
        quintile_bucket, bins = pd.qcut(
            yhat_test, 5, labels=False, retbins=True, duplicates="drop"
        )
        quintile_labels = [f"Q{i}" for i in range(1, len(bins))]
        quintile_bucket = pd.cut(
            yhat_test, bins=bins, labels=quintile_labels, include_lowest=True
        )

        quintile_df["model"] = (
            pd.DataFrame({"y": yhat_test, "quintile_bucket": quintile_bucket})
            .groupby("quintile_bucket")["y"]
            .mean()
            .reindex(quintile_labels)
            .to_numpy()
        )

        # Plot a line for the naive model
        p.line(
            quintile_df["quintile"],
            quintile_df["naive_model"],
            legend_label=f"Naive Model: Fold {fold}",
            line_width=1,
            line_color="green",
        )

        # Plot a line for the model predictions
        p.line(
            quintile_df["quintile"],
            quintile_df["model"],
            legend_label=f"Model: Fold {fold}",
            line_width=1,
            line_color="firebrick",
        )

        # Plot a vertical bar for the actual values
        p.vbar(
            x="quintile",
            top="mean_target",
            source=quintile_df,
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
    col1, col2 = two_column_layout_with_spacers()
    if feature_type == "continuous" and target_type in ["categorical", "binary"]:
        with col1:
            density_plot(X, y, fold)

        with col2:
            st.markdown("### Density Plot")

    elif feature_type in ["categorical", "binary"] and target_type in [
        "categorical",
        "binary",
    ]:
        with col1:
            stacked_bar_chart(X, y)

        with col2:
            st.markdown("### Stacked Bar Chart")

    roc_curve(df, univariate_feature_variable, target_variable)

    lift_plot(X, y, fold)


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