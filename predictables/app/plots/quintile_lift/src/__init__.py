from __future__ import annotations

import pandas as pd
import streamlit as st

from bokeh.plotting import figure
from predictables.app import train_test_split
from predictables.app.plots.quintile_lift.src.rf_model import rf_model
from predictables.app.plots.quintile_lift.src.quintile_lift_df import quintile_lift_df


def quintile_lift(
    X: pd.Series, y: pd.Series, folds: pd.Series
) -> tuple[figure, pd.DataFrame]:
    """Generate a quintile lift plot for the univariate analysis."""
    p = figure(
        title="Quintile Lift Plot",
        x_axis_label="Quintile",
        y_axis_label="Lift",
        width=1000,
        height=600,
    )

    df_list = []
    fold_list = folds[folds > 0].drop_duplicates().sort_values().tolist()
    for fold in fold_list:
        X_train, X_test, y_train, y_test = train_test_split(X, y, folds, fold, True)

        model = rf_model(X_train, y_train)
        quintile_df = quintile_lift_df(y_train, X_test, y_test, model)

        df_list.append(quintile_df.assign(fold=fold))

        # Plot a line for the naive model
        p.line(
            quintile_df["quintile"],
            quintile_df["naive_model"],
            line_width=1,
            line_alpha=0.5,
            line_color="green",
        )

        # Plot a line for the model predictions
        p.line(
            quintile_df["quintile"],
            quintile_df["random_forest_model"],
            line_width=1,
            line_alpha=0.5,
            line_color="firebrick",
        )

    df = (
        pd.concat(df_list)[["quintile", "actual", "naive_model", "random_forest_model"]]
        .groupby("quintile")
        .mean()
        .reset_index()
    )

    # Plot a vertical bar for the actual values

    p.vbar(
        x=df["quintile"],
        top=df["actual"],
        width=0.5,
        legend_label="Actual",
        fill_alpha=0.3,
        line_color="black",
        fill_color="skyblue",
    )

    p.line(
        x=df["quintile"],
        y=df["random_forest_model"],
        line_width=3,
        line_color="firebrick",
        legend_label="Random Forest Model",
    )

    p.line(
        x=df["quintile"],
        y=df["naive_model"],
        line_width=3,
        line_color="green",
        legend_label="Naive Model",
    )

    p.legend.location = "bottom_right"
    p.legend.click_policy = "hide"

    return p, df