from PredicTables.univariate.src.plots.util.plot_label import plot_label
from PredicTables.util import get_column_dtype

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import chi2_contingency, entropy, mannwhitneyu, norm, ttest_ind


def density_plot(backend="matplotlib", **kwargs):
    if backend == "matplotlib":
        density_plot_mpl(**kwargs)
    elif backend == "plotly":
        density_plot_plotly(**kwargs)
    else:
        raise ValueError("backend must be either matplotlib or plotly")


def density_plot_mpl(
    feature, target, is_target_binary=True, cv_idx=None, ax=None, cv_alpha=0.2, figsize=(10, 6)
):
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    data = pd.DataFrame({"feature": feature, "target": target})

    if 
        data[feature] = data[feature].astype("category")

    if get_column_dtype(target) == "binary":
        data[target] = data[target].astype("category")

    # Plot the violin plot
    ax = violin_plot(
        target=target,
        feature=feature,
        outlier_df=outlier_df,
        cv_folds_data=cv_idx,
        cv_alpha=cv_alpha,
        ax=ax,
        dropvals=[-0.01, -1],
    )

    return ax


def density_plot_plotly(cv_alpha=0.2):
    data = GetTrain()[[feature, target]]

    if type == "categorical":
        data[feature] = data[feature].astype("category")

    # Prepare the data
    data[feature] = data[feature].cat.remove_unused_categories()
    categories = data[feature].cat.categories

    # Initialize figure
    fig = go.Figure()

    # Plot the violin plot
    for category in categories:
        category_data = data[data[feature] == category]
        fig.add_trace(
            go.Violin(
                y=category_data[target],
                name=str(category),
                box_visible=True,
                meanline_visible=True,
                opacity=cv_alpha,
                points="outliers",  # or 'all' or False
            )
        )

    # Customize layout
    fig.update_layout(
        title=f"Lift Plot - Model Including [{_plot_label(feature)}] vs Null Model",
        yaxis_title=_plot_label(target),
        xaxis_title=_plot_label(feature),
    )

    return fig


def plot_density_plot(self, significance_level=0.05, ax=None, opacity=0.5):
    df = self.train[[self.feature, self.target]]

    if ax is None:
        _, ax = plt.subplots(figsize=self.figsize)

    unique_targets = df[self.target].unique()

    data_by_target = {}
    for target_val in unique_targets:
        sns.kdeplot(
            df[df[self.target] == target_val][self.feature],
            ax=ax,
            label=f"{self._plot_label(self.target)} = {target_val}",
            alpha=opacity,
            fill=True,
        )
        data_by_target[target_val] = df[df[self.target] == target_val][self.feature]

    # Mann-Whitney U Test
    u_stat, p_value = mannwhitneyu(
        data_by_target[unique_targets[0]], data_by_target[unique_targets[1]]
    )

    # Print the results of the Mann-Whitney U Test
    subtitle_message = (
        f"\nDistributions are the same at the {significance_level:.0%} level"
        if p_value > significance_level
        else f"\nDistributions are different at the {significance_level:.0%} level"
    )

    ax.set_title(
        f"Density Plot of [{self._plot_label(self.feature)}] by \
[{self._plot_label(self.target)}]{subtitle_message}"
    )

    annotation_text = f"Mann-Whitney\nU-Test Statistic:\n{u_stat:.1f}\n\np-value:\n\
{p_value:.2f}"
    ax.annotate(
        annotation_text,
        xy=(0.7375, 0.625),
        xycoords="axes fraction",
        fontsize=16,
        bbox=dict(
            boxstyle="round,pad=0.3",
            edgecolor="black",
            facecolor="aliceblue",
            alpha=0.5,
        ),
    )

    ax.set_xlabel(self._plot_label(self.feature))
    ax.set_ylabel("Density")
    ax.legend(loc="upper right", fontsize=16)

    ax = rotate_x_lab(ax)
    ax.figure.tight_layout()
    return ax

def plotly_density_plot(self, significance_level=0.05, opacity=0.5):
    df = self.train[[self.feature, self.target]]
    unique_targets = df[self.target].unique()

    fig = go.Figure()

    data_by_target = {}
    for target_val in unique_targets:
        # Filter the data for the current target value
        filtered_data = df[df[self.target] == target_val][self.feature]
        data_by_target[target_val] = filtered_data

        # Add a KDE trace for the current target value
        fig.add_trace(
            go.Histogram(
                x=filtered_data,
                histnorm="probability density",
                opacity=opacity,
                name=f"{self._plot_label(self.target)} = {target_val}",
            )
        )

    # Perform Mann-Whitney U Test between the two groups
    u_stat, p_value = mannwhitneyu(
        data_by_target[unique_targets[0]], data_by_target[unique_targets[1]]
    )

    # Determine the subtitle message based on p_value
    subtitle_message = (
        f"Distributions are the same at the {significance_level:.0%} significance level"
        if p_value > significance_level
        else f"Distributions are different at the {significance_level:.0%} significance level"
    )

    # Add annotations with the Mann-Whitney U Test results
    fig.add_annotation(
        xref="paper",
        yref="paper",
        x=0.95,
        y=0.95,
        text=f"Mann-Whitney U-Test Statistic: {u_stat:.1f}<br>p-value: {p_value:.2f}",
        showarrow=False,
        font=dict(size=16),
        align="right",
        bgcolor="aliceblue",
        opacity=0.8,
        bordercolor="black",
        borderwidth=1,
        borderpad=4,
    )

    # Update the layout of the figure
    fig.update_layout(
        title=f"Density Plot of {self._plot_label(self.feature)} by {self._plot_label(self.target)}<br><sup>{subtitle_message}</sup>",
        xaxis_title=self._plot_label(self.feature),
        yaxis_title="Density",
        barmode="overlay",  # Overlay the KDE plots
        legend_title_text="Legend",
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
        ),
    )

    return fig