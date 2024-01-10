import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns

from PredicTables.util import to_pd_df


def plot_initial_densities(df, grid_size=(3, 3)):
    """
    Plot the density of the non-missing values in a 3x3 grid.

    :param df: DataFrame with numeric columns
    :param grid_size: Size of the grid for plotting
    """
    df = to_pd_df(df)

    fig, axes = plt.subplots(grid_size[0], grid_size[1], figsize=(20, 20))
    axes = axes.flatten()  # Flatten the grid for easy indexing

    for idx, col in enumerate(df.columns):
        # Plot density curve for each column
        sns.kdeplot(
            df[col].dropna(),
            ax=axes[idx],
            linestyle="--",
            lw=2,
            label=f"Original data distribution for {col}",
        )
        axes[idx].set_title(col)
        axes[idx].legend()
        axes[idx].set_ylim(0, 1)

    plt.tight_layout()
    plt.show()


def plot_density_with_histograms(
    original_df, imputed_df, new_imputed_df=None, grid_size=(3, 3)
):
    """
    Plot the original density curves and overlay histograms of imputed values.

    :param original_df: Original DataFrame with missing values
    :param imputed_df: DataFrame after imputation
    :param grid_size: Size of the grid for plotting
    """
    original_df = to_pd_df(original_df)
    imputed_df = to_pd_df(imputed_df)

    num_plots = grid_size[0] * grid_size[1]
    selected_columns = original_df.columns[:num_plots]

    _, axes = plt.subplots(nrows=grid_size[0], ncols=grid_size[1], figsize=(20, 20))
    axes = axes.flatten()

    for idx, col in enumerate(selected_columns):
        # Plot original density curve
        sns.kdeplot(
            original_df[col].dropna(),
            ax=axes[idx],
            linestyle="--",
            lw=2,
            color="blue",
            label=f"Original data distribution for {col}",
        )
        # Overlay histogram of imputed values
        axes[idx].hist(
            imputed_df[col],
            bins=30,
            alpha=0.25,
            color="red",
            density=True,
            label=f"Imputed data distribution for {col}",
        )
        if new_imputed_df is not None:
            axes[idx].hist(
                new_imputed_df[col],
                bins=30,
                alpha=0.25,
                color="blue",
                density=True,
                label=f"New imputed data distribution for {col}",
            )
        axes[idx].set_title(col)
        axes[idx].legend()
        axes[idx].set_ylim(0, 1)

    for idx in range(len(selected_columns), num_plots):
        axes[idx].set_visible(False)

    plt.tight_layout()
    plt.show()


def preprocess_data(df):
    """
    Preprocess the DataFrame for visualization.
    Identifies numeric columns with missing values and returns them.

    :param df: DataFrame to be processed
    :return: DataFrame with numeric columns having missing values
    """
    df = to_pd_df(df)
    # Identify numeric columns with missing values
    numeric_cols = df.select_dtypes(include=[np.number])
    cols_with_missing = numeric_cols.columns[numeric_cols.isnull().any()]

    return df[cols_with_missing]


def rmse(actual, predicted):
    return np.sqrt(np.mean((actual - predicted) ** 2))


def get_rmse(df1, df2, col):
    df1 = to_pd_df(df1)
    df2 = to_pd_df(df2)
    actual = df1[col]
    predicted = df2[col]
    return rmse(actual.values, predicted.values)


def get_mse(act, pred):
    return (
        pl.DataFrame({"pred": pred, "act": act})
        .lazy()
        .filter(pl.col("pred").is_not_null())
        .filter(pl.col("pred").is_not_nan())
        .filter(~pl.col("pred").is_infinite())
        .with_columns([(pl.col("act") - pl.col("pred")).pow(2).alias("mse")])
        .select([pl.col("mse").mean().name.keep()])
        .collect()
        .item()
    )


def plot_scatter_matrix(
    original_df,
    init_impute_df,
    imputed_df,
    bins=40,
    n_features=4,
    figsize=(20, 20),
    actual_df=None,
    missing_cols=None,
    val=None,
    imputed_val=None,
):
    original_df = to_pd_df(original_df)
    init_impute_df = to_pd_df(init_impute_df)
    imputed_df = to_pd_df(imputed_df)

    if missing_cols is not None:
        original_df = original_df[missing_cols]
        init_impute_df = init_impute_df[missing_cols]
        imputed_df = imputed_df[missing_cols]

        if actual_df is not None:
            actual_df = actual_df[missing_cols]

    fig, ax = plt.subplots(n_features, n_features, figsize=figsize)
    for i in range(n_features):
        for j in range(n_features):
            col = original_df.columns.tolist()[i]
            if i != j:
                # regular scatter plot
                original_df.iloc[:, [i, j]].plot(
                    kind="scatter",
                    x=original_df.columns[i],
                    y=original_df.columns[j],
                    ax=ax[i, j],
                    grid=True,
                    marker="o",
                    color="black",
                    alpha=0.15,
                    edgecolors="none",
                    label="Original Data",
                )

                missing_mask = original_df.isnull().iloc[:, [i, j]]
                init_impute = init_impute_df.iloc[:, [i, j]].loc[
                    missing_mask.iloc[:, 0].values | missing_mask.iloc[:, 1].values
                ]
                imputed = imputed_df.iloc[:, [i, j]].loc[
                    missing_mask.iloc[:, 0].values | missing_mask.iloc[:, 1].values
                ]
                if i < j:
                    ax[i, j].scatter(
                        init_impute.iloc[:, 0],
                        init_impute.iloc[:, 1],
                        marker=".",
                        color="red",
                        label="Initial Imputation",
                        alpha=0.5,
                        edgecolors="none",
                    )
                else:
                    ax[i, j].scatter(
                        imputed.iloc[:, 0],
                        imputed.iloc[:, 1],
                        marker=".",
                        color="blue",
                        label="Final Imputation",
                        alpha=0.5,
                        edgecolors="none",
                    )

                ax[i, j].set_xlim(
                    original_df.iloc[:, i].min(), original_df.iloc[:, i].max()
                )
                ax[i, j].set_ylim(
                    original_df.iloc[:, j].min(), original_df.iloc[:, j].max()
                )
                ax[i, j].legend(loc="upper right")

            else:
                missing_mask = original_df.isnull().iloc[:, [i, j]]
                original_df.iloc[:, i].plot(
                    kind="kde",
                    ax=ax[i, j],
                    grid=True,
                    color="black",
                    ls="--",
                    lw=2,
                    label="Original Data",
                )
                ax[i, j].hist(
                    init_impute_df.iloc[:, i],
                    color="red",
                    alpha=0.3,
                    label="Initial Imputation",
                    histtype="stepfilled",
                    edgecolor="none",
                    density=True,
                    bins=bins,
                )
                ax[i, j].hist(
                    imputed_df.iloc[:, i],
                    color="blue",
                    alpha=0.3,
                    label="Final Imputation",
                    histtype="stepfilled",
                    edgecolor="none",
                    density=True,
                    bins=bins,
                )
                ax[i, j].legend(loc="upper right")
                ax[i, j].set_title(
                    f"Validation Set MSE: {get_mse(val[col], imputed_val[col]):.3f}",
                    fontsize=14,
                )

    plt.show()
