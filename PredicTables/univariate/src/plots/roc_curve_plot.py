from typing import Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from sklearn.metrics import RocCurveDisplay, roc_auc_score, roc_curve

# Here's an outline of the refactoring plan:

#     Data Preparation for Plotting: Isolating the logic that prepares the data for the ROC curve plot.
#     Plotting Individual ROC Curves: Creating a function to plot the ROC curve for each fold.
#     Calculating Mean and Standard Deviation: Computing mean and standard deviation of the True Positive Rates (TPR) across all folds.
#     Plotting Mean ROC Curve and Confidence Band: A function to plot the mean ROC curve with the confidence band.
#     Annotation and Legend Configuration: Separating the code for annotating the plot and configuring the legend.
#     Statistical Analysis and Annotation: Including DeLong test and other statistical annotations on the plot.
#     Model Fit Information Annotation: A function to annotate the plot with logistic regression model fit information.
#     Final Plot Adjustments and Rendering: Adjusting final plot settings like axis labels, title, and layout.
#     Main Function Integration: Integrating all these components in the main plot_cv_roc_auc function.


# def roc_curve_plot(
#     y: pd.Series, yhat: pd.Series, backend: str = "matplotlib", **kwargs
# ):
#     """
#     Plot the ROC curve for a single model.

#     Parameters
#     ----------
#     y : pd.Series
#         The true labels.
#     yhat : pd.Series
#         The predicted labels.
#     backend : str
#         The plotting backend to use. Either "matplotlib" or "plotly". Defaults to "matplotlib".
#     **kwargs
#         Additional keyword arguments to pass to the plotting function.

#     Returns
#     -------
#     figax : Union[go.Figure, Axes]
#         The plot.
#     """
#     if backend == "plotly":
#         # return plot_cv_roc_auc_plotly(y, yhat, **kwargs)
#         raise NotImplementedError("Plotly backend not implemented yet.")
#     else:
#         return plot_cv_roc_auc_mpl(y, yhat, **kwargs)


def create_auc_data(
    y: pd.Series, yhat_proba: pd.Series, n_bins: int = 200
) -> Tuple[pd.Series, pd.Series]:
    """
    Create the data for plotting an ROC curve. Calculates the false positive rate
    at each threshold and the true positive rate at each threshold.

    Parameters
    ----------
    y : pd.Series
        The true labels.
    yhat_proba : pd.Series
        The predicted probabilities.
    n_bins : int
        The number of bins to use when calculating the ROC curve. Generally, the
        more bins, the smoother the curve. Defaults to 200.

    Returns
    -------
    fpr : pd.Series
        The false positive rate, ranging from 0 to 1, at each threshold.
    tpr : pd.Series
        The true positive rate, ranging from 0 to 1, at each threshold.
    """
    roc = roc_curve(y, yhat_proba)

    # Interpolate the data to get a smoother curve
    fpr = pd.Series(np.linspace(0, 1, n_bins))
    tpr = pd.Series(np.interp(fpr, roc[0], roc[1]))

    return fpr, tpr


def plot_individual_roc_curves(
    y: pd.Series,
    yhat_proba: pd.DataFrame,
    curve_name: str = "ROC Curve",
    figax: Union[go.Figure, Axes, None] = None,
    n_bins: int = 200,
    alpha: float = 0.4,
    legendgroup: Union[str, None] = None,
    figsize: Tuple[int, int] = (15, 7),
    ax: Axes = None,
    fig: go.Figure = None,
    plot_title: str = "ROC Curve (AUC = [AUC])\nROC AUC is [significance_stmnt]",
    backend: str = "matplotlib",
) -> Union[go.Figure, Axes]:
    """
    Plot the ROC curve for each fold.

    Parameters
    ----------
    y : pd.Series
        The true labels.
    yhat_proba : pd.DataFrame
        The predicted probabilities.
    curve_name : str, optional
        The name of the curve to be plotted. If not provided, defaults to "ROC Curve".
    figax : Union[go.Figure, Axes, None], optional
        The plot. Will be ignored if either `fig` (in the case of plotly) or `ax` (in the case of matplotlib) is provided.
    n_bins : int, optional
        The number of bins to use when calculating the ROC curve. Generally, the
        more bins, the smoother the curve. Defaults to 200.
    alpha : float, optional
        The opacity of the individual ROC curves.
    legendgroup : Union[str, None], optional
        The legend group to use for the individual ROC curves. If None, no legend group is used.
    figsize : Tuple[int, int], optional
        The figure size to use. Defaults to (15, 7).
    ax : matplotlib.axes.Axes, optional
        Alias for figax. If provided and the backend is "matplotlib", figax is ignored.
    fig : plotly.graph_objects.Figure, optional
        Alias for figax if using plotly. If provided and the backend is "plotly", figax is ignored.
    plot_title : str, optional
        The title of the plot. Defaults to:
            ROC Curve (AUC = [AUC])
            ROC AUC is [significance_stmnt].
        This will be formatted with the AUC and a statement regarding the significance of the AUC estimate.
    backend : str, optional
        The plotting backend to use. Either "matplotlib" or "plotly". Defaults to "matplotlib".

    Returns
    -------
    figax : Union[go.Figure, Axes]
        The plot.
    """
    fpr, tpr = create_auc_data(y, yhat_proba, n_bins)

    # Handle the case where figax is an alias for either fig or ax
    if (backend == "matplotlib") and (ax is not None):
        figax = ax
    elif (backend == "plotly") and (fig is not None):
        figax = fig
    else:
        pass

    if backend == "plotly":
        if isinstance(figax, go.Figure) or (figax is None):
            figax.add_trace(
                go.Scatter(
                    x=fpr,
                    y=tpr,
                    mode="lines",
                    hoverdata=pd.DataFrame(dict(fpr=fpr, tpr=tpr)),
                    hovertemplate=f"<h4>{curve_name}</h4><br>FPR: {fpr:.3f}<br>TPR: {tpr:.3f}<extra></extra>",
                    legendgroup="Folds",
                    opacity=alpha,
                    line=dict(dash="dot"),
                    showlegend=False,
                )
                if legendgroup is not None
                else go.Scatter(
                    x=fpr,
                    y=tpr,
                    mode="lines",
                    hoverdata=pd.DataFrame(dict(fpr=fpr, tpr=tpr)),
                    hovertemplate=f"<h4>{curve_name}</h4><br>FPR: {fpr:.3f}<br>TPR: {tpr:.3f}<extra></extra>",
                    opacity=alpha,
                    line=dict(dash="dot"),
                    showlegend=False,
                )
            )

            # Update the layout for the figsize
            figax.update_layout(
                width=figsize[0],
                height=figsize[1],
            )
            return figax
        else:
            raise TypeError(
                "figax must be a plotly.graph_objects.Figure object when using the plotly backend"
            )
    elif backend == "matplotlib":
        if isinstance(figax, Axes):
            RocCurveDisplay(
                fpr=fpr,
                tpr=tpr,
            ).plot(ax=figax, alpha=alpha, label=curve_name)

            # Update the figure size for the figsize
            figax.figure.set_size_inches(figsize[0], figsize[1])

            return figax
        elif figax is None:
            _, ax = plt.subplots(figsize=figsize)
            RocCurveDisplay(
                fpr=fpr,
                tpr=tpr,
            ).plot(ax=ax, alpha=alpha, label=curve_name)

            return ax

        else:
            raise TypeError(
                "figax must be a matplotlib.axes.Axes object when using the matplotlib backend"
            )

    else:
        raise ValueError(
            f"Invalid backend (expecting either 'matplotlib' or 'plotly'): {backend}"
        )


def plot_cv_roc_curves(
    y: pd.Series,
    yhat_proba: pd.DataFrame,
    fold: pd.Series,
    figax: Union[go.Figure, Axes, None] = None,
    n_bins: int = 200,
    cv_alpha: float = 0.4,
    ax: Axes = None,
    fig: go.Figure = None,
    backend: str = "matplotlib",
) -> Union[go.Figure, Axes]:
    """
    Plot the ROC curve for each fold. Filters the data for each fold label,
    then plots the ROC curve for a model trained on that fold.

    Parameters
    ----------
    y : pd.Series
        The true labels.
    yhat_proba : pd.DataFrame
        The predicted probabilities.
    fold : pd.Series
        The fold number for each observation.
    figax : Union[go.Figure, Axes, None], optional
        The plot.
    n_bins : int, optional
        The number of bins to use when calculating the ROC curve. Generally, the
        more bins, the smoother the curve. Defaults to 200.
    cv_alpha : float, optional
        The opacity of the individual ROC curves.
    ax : matplotlib.axes.Axes, optional
        Alias for figax. If provided and the backend is "matplotlib", figax is ignored.
    fig : plotly.graph_objects.Figure, optional
        Alias for figax if using plotly. If provided and the backend is "plotly", figax is ignored.
    backend : str, optional
        The plotting backend to use. Either "matplotlib" or "plotly". Defaults to "matplotlib".

    Returns
    -------
    figax : Union[go.Figure, Axes]
        The plot.
    """
    if (backend == "matplotlib") and (ax is not None):
        figax = ax
    elif (backend == "plotly") and (fig is not None):
        figax = fig
    elif (backend == "matplotlib") and (figax is None) and (ax is None):
        _, ax = plt.subplots()
        figax = ax

    for f in fold.drop_duplicates().sort_values().values:
        figax = plot_individual_roc_curves(
            y=y[fold == f],
            yhat_proba=yhat_proba[fold == f],
            curve_name=f"Fold {f}",
            figax=figax,
            n_bins=n_bins,
            alpha=cv_alpha,
            legendgroup="Folds",
            backend=backend,
        )

    return figax


def calc_auc_curve_data_from_folds(
    y: pd.Series, yhat_proba: pd.DataFrame, fold: pd.Series, n_bins: int = 200
):
    """
    Calculate the standard error of the ROC curve for each fold. Filters the data for each fold label,
    then plots the ROC curve for a model trained on that fold.

    Parameters
    ----------
    y : pd.Series
        The true labels.
    yhat_proba : pd.DataFrame
        The predicted probabilities.
    fold : pd.Series
        The fold number for each observation.
    n_bins : int, optional
        The number of bins to use when calculating the ROC curve. Generally, the
        more bins, the smoother the curve. Defaults to 200.

    Returns
    -------
    se_fpr : pd.Series
        The standard error of the false positive rate, ranging from 0 to 1, at each threshold.
    se_tpr : pd.Series
        The standard error of the true positive rate, ranging from 0 to 1, at each threshold.
    """
    # Data preparation

    # Calculate the standard error of the ROC curve for each fold
    fprs = pd.DataFrame()
    tprs = pd.DataFrame()
    for f in fold.drop_duplicates().sort_values().values:
        fpr, tpr = create_auc_data(y[fold == f], yhat_proba[fold == f], n_bins)
        fprs[f"fold_{f}"] = fpr
        tprs[f"fold_{f}"] = tpr

    total_fpr, total_tpr = create_auc_data(y, yhat_proba, n_bins)
    fprs["mean"] = total_fpr
    tprs["mean"] = total_tpr

    se_fpr = (
        fprs.apply(lambda x: np.power(x - x["mean"], 2), axis=1)
        .drop(columns="mean")
        .mean(axis=1)
        .apply(np.sqrt)
    )
    se_tpr = (
        tprs.apply(lambda x: np.power(x - x["mean"], 2), axis=1)
        .drop(columns="mean")
        .mean(axis=1)
        .apply(np.sqrt)
    )

    fprs["se0"] = se_fpr
    tprs["se0"] = se_tpr

    # apply a gaussian filter to smooth the standard error -- otherwise it can
    # change dramatically from one bin (eg 50 basis points) to the next

    fprs["se"] = fprs["se0"].rolling(5, center=True).mean().fillna(fprs["se0"])
    tprs["se"] = tprs["se0"].rolling(5, center=True).mean().fillna(tprs["se0"])

    fprs["mean+1"] = fprs["mean"] + fprs["se"]
    fprs["mean-1"] = fprs["mean"] - fprs["se"]
    tprs["mean+1"] = tprs["mean"] + tprs["se"]
    tprs["mean-1"] = tprs["mean"] - tprs["se"]

    return fprs, tprs


def plot_roc_auc_curves_and_confidence_bands(
    y: pd.Series,
    yhat_proba: pd.DataFrame,
    fold: pd.Series,
    figax: Union[go.Figure, Axes, None] = None,
    n_bins: int = 200,
    cv_alpha: float = 0.4,
    ax: Axes = None,
    fig: go.Figure = None,
    figsize: Tuple[int, int] = (8, 8),
    backend: str = "matplotlib",
) -> Union[go.Figure, Axes]:
    if backend == "plotly":
        raise NotImplementedError("Plotly backend not implemented yet.")
    elif backend == "matplotlib":
        if ax is not None:
            figax = ax
        elif (ax is None) and (figax is None):
            _, figax = plt.subplots(figsize=figsize)
        elif backend == "matplotlib":
            pass
        else:
            raise TypeError(
                "figax must be a matplotlib.axes.Axes object when using the matplotlib backend"
            )

        fprs, tprs = calc_auc_curve_data_from_folds(y, yhat_proba, fold, n_bins)

        for col in fprs.columns:
            if col.startswith("fold"):
                figax.plot(
                    fprs[col],
                    tprs[col],
                    alpha=cv_alpha,
                    label="_" + col.replace("_", " ").title(),
                    lw=0.5,
                )
            elif col == "mean":
                figax.plot(
                    fprs[col],
                    tprs[col],
                    alpha=1,
                    label=col.replace("_", " ").title(),
                    lw=2,
                )
            elif col.startswith("mean"):
                figax.plot(
                    fprs[col],
                    tprs[col],
                    alpha=1,
                    label="_"
                    + col.replace("+", " + ").replace("-", " - ").title()
                    + "SD",
                    lw=0.5,
                    ls="--",
                    color="grey",
                )

        figax.fill_between(
            fprs["mean"],
            tprs["mean+1"],
            tprs["mean-1"],
            alpha=0.2,
            color="grey",
            label="Mean(ROC) +/- 1 SD(ROC)",
        )

        figax.plot(
            [0, 1], [0, 1], color="navy", lw=1, linestyle="--", label="Random Guess"
        )

        plt.legend()

        return figax


def delong_statistic_annotation(y: pd.Series, yhat_proba: pd.Series, ax: Axes):
    """
    Implement the DeLong test to compare the ROC AUC against the 45-degree
    line (AUC=0.5).

    The DeLong test uses the Central Limit Theorem (CLT) to approximate
    the distribution of the AUC as normal. The test computes the covariance
    matrix of the paired AUC differences and uses it to generate a Z-statistic.
    According to CLT, this Z-statistic will be approximately normally distributed
    for sufficiently large sample sizes (typically n > 30).

    Parameters
    ----------
    y : pd.Series
        The true labels.
    yhat_proba : pd.Series
        The predicted probabilities.
    ax : Axes
        The Axes object to be configured.

    Returns
    -------
    ax : matplotlib.Axes.Axes
        The Axes object annotated with the DeLong test statistic and p-value.
    """
    z, p = _delong_test_against_chance(y, yhat_proba)

    significance_message = (
        f"DeLong Test Statistic\nAgainst the 45-degree Line:\n\nz = {z:.3f}\np-value = {p:.3f}"
    )
    significance_message += "\n" + (
        "\nThe indicated AUC is\nsignificantly different\nfrom random guessing at \nthe 95\% confidence level."
        if p < 0.05
        else "The indicated AUC is not significantly different from random guessing at the 95\% confidence level."
    )

    # add annotation
    ax.annotate(
        significance_message,
        xy=(0.6, 0.5),
        xycoords="axes fraction",
        fontsize=12,
        bbox=dict(
            boxstyle="round,pad=0.3", edgecolor="black", facecolor="white", alpha=0.5
        ),
    )

    return ax


def add_statistical_analysis_annotation_mpl(
    ax: Axes, dl_stat, p_value, significance_level
):
    # Annotate the plot with hypothesis testing information
    significance_message = f"DeLong Test Statistic Against the 45-degree Line: {dl_stat:.3f}, p-value = {p_value:.3f}"
    ax.annotate(
        significance_message,
        xy=(0.6, 0.6),
        xycoords="Axes fraction",
        fontsize=12,
        bbox=dict(
            boxstyle="round,pad=0.3", edgecolor="black", facecolor="white", alpha=0.5
        ),
    )

    # Additional message based on significance level
    if p_value < significance_level:
        ax.annotate(
            "Significantly different from random",
            xy=(0.6, 0.55),
            xycoords="Axes fraction",
            fontsize=10,
        )
    else:
        ax.annotate(
            "Not significantly different from random",
            xy=(0.6, 0.55),
            xycoords="Axes fraction",
            fontsize=10,
        )

    return ax


def add_model_fit_annotation_mpl(
    ax: Axes,
    coef: float,
    std_error: float,
    pvalue: float,
    alpha: float = 0.05,
):
    from scipy.stats import norm

    z_alpha = norm.ppf(1 - alpha / 2)
    ci_lower = coef - z_alpha * std_error
    ci_upper = coef + z_alpha * std_error
    pct = f"{1 - alpha:.1%}"

    significance_stmnt = (
        f"Coefficient is significantly different from 0 at the {pct} level"
        if pvalue < alpha
        else f"Coefficient is not significantly different from 0 at the {pct} level"
    )

    # Annotation text
    text = (
        f"Estimated Coefficient: {coef:.2f}\n"
        f"{pct} Confidence Interval: [{ci_lower:.2f}, {ci_upper:.2f}]\n"
        f"p-value: {pvalue:.2f}\n"
        f"{significance_stmnt}"
    )

    # Add annotation to the plot
    ax.annotate(
        text,
        xy=(0.05, 0.05),
        xycoords="Axes fraction",
        fontsize=12,
        bbox=dict(
            boxstyle="round,pad=0.3", edgecolor="black", facecolor="white", alpha=0.5
        ),
    )

    return ax


def auc(y: pd.Series, yhat: pd.Series) -> float:
    """
    Compute the area under the ROC curve.

    Parameters
    ----------
    y : pd.Series
        The true labels.
    yhat : pd.Series
        The predicted labels.

    Returns
    -------
    auc : float
        The area under the ROC curve.
    """
    from sklearn.metrics import roc_auc_score

    return roc_auc_score(y, yhat)


def calculate_interpolated_mean_std_tpr(
    mean_fpr: pd.Series, tprs: pd.DataFrame
) -> Tuple[pd.Series, pd.Series]:
    """
    Calculate the mean and standard deviation of the TPR (true positive rate)
    across all folds.

    Parameters
    ----------
    mean_fpr : pd.Series
        The mean false positive rate across all folds.
    tprs : pd.DataFrame
        The true positive rates across all folds.

    Returns
    -------
    mean_tpr : pd.Series
        The mean true positive rate across all folds.
    """
    mean_tpr, std_tpr = calculate_mean_std_tpr(tprs)

    # Interpolate mean_tpr to match the length of mean_fpr
    mean_tpr_interpolated = np.interp(
        mean_fpr, np.linspace(0, 1, len(mean_tpr)), mean_tpr
    )

    return mean_tpr_interpolated, std_tpr


def finalize_plot(ax: Axes, figsize: Tuple[int, int]) -> Axes:
    """
    Finalize the plot by adjusting the figure size and layout.

    Parameters
    ----------
    ax : Axes
        The Axes object to be configured.
    figsize : Tuple[int, int]
        The figure size to use.

    Returns
    -------
    ax : matplotlib.Axes.Axes
        The configured Axes object.
    """
    # Adjust figure size and layout
    plt.gcf().set_size_inches(figsize)
    plt.tight_layout()

    return ax


# def plot_cv_roc_auc_mpl(
#     n_bins: int,
#     y: pd.Series,
#     yhat_proba: pd.Series,
#     fold: Union[pd.Series, None] = None,
#     alpha: float = 0.05,
#     figsize: Tuple[int, int] = (10, 6),
#     coef: float = None,
#     std_error: float = None,
#     pvalue: float = None,
# ) -> Axes:
#     # Data preparation
#     fpr, tpr = create_auc_data(y, yhat, n_bins)

#     # Plot creation
#     _, ax = plt.subplots()

#     # Plot individual ROC curves
#     if fold is not None:
#         for f in fold.drop_duplicates().sort_values().values:
#             plot_individual_roc_curves(
#                 ax=ax,
#                 y=y[fold == f],
#                 yhat_proba=yhat_proba[fold == f],
#                 curve_name=f"Fold {f}",
#                 alpha=alpha,
#                 legendgroup="Folds",
#             )
#     plot_individual_roc_curves(ax, fpr, tprs)

#     # Calculate and plot mean ROC curve and confidence band
#     mean_tpr, std_tpr = calculate_interpolated_mean_std_tpr(fpr, tprs)
#     mean_auc = auc(fpr, mean_tpr)
#     plot_mean_roc_and_confidence_band_mpl(ax, fpr, mean_tpr, std_tpr, mean_auc)

#     # Configure annotations and legend
#     configure_annotations_and_legend_mpl(ax, mean_auc, std_tpr)

#     # Statistical analysis and annotation
#     (dl_stat, p_value) = _delong_test_against_chance(y, yhat)
#     add_statistical_analysis_annotation_mpl(ax, dl_stat, p_value, alpha)

#     # Model fit information annotation
#     add_model_fit_annotation_mpl(ax=ax, coef=coef, std_error=std_error, pvalue=pvalue)

#     # Finalize the plot
#     ax = finalize_plot(ax, figsize)

#     return ax


def _compute_auc_variance(y: pd.Series, yhat: pd.Series):
    """
    Compute the variance of the AUC estimator. This is used in the
    computation of the DeLong test, and is based on the following paper:

    @article{delong1988comparing,
    title={Comparing the areas under two or more correlated receiver operating characteristic curves: a nonparametric approach},
    author={DeLong, Elizabeth R and DeLong, David M and Clarke-Pearson, Daniel L},
    journal={Biometrics},
    pages={837--845},
    year={1988},
    publisher={JSTOR}
    }

    Var(AUC) = (V1 + V2 + V3) / (n1 * n0)

    where
    - V1 = AUC * (1 - AUC)
    - V2 = (n1 - 1) * (Q1 - AUC^2)
    - V3 = (n0 - 1) * (Q0 - AUC^2)
    - Q1 = AUC / (2 - AUC)
    - Q0 = (2 * AUC^2) / (1 + AUC)
    - n1 = number of positive classes
    - n0 = number of negative classes

    Parameters
    ----------
    None. Relies on the following class attributes:
        - fitted: dictionary of fitted models and metrics
        - target: name of the target variable

    Returns
    -------
    var_auc : variance of the AUC estimator
    """
    auc = roc_auc_score(y, yhat)
    auc2 = np.power(auc, 2)

    # Count of positive and negative classes
    n = y.shape[0]
    n1 = y[y == 1].sum()
    n0 = n - n1

    # Q1 and Q2 for variance calculation
    Q1 = auc / np.subtract(2, auc)
    Q0 = np.divide((2 * auc2), np.add(1, auc))

    # Compute the variance
    var_auc = (
        auc * np.subtract(1, auc) + (n1 - 1) * (Q1 - auc2) + (n0 - 1) * (Q0 - auc2)
    ) / (n1 * n0)

    return var_auc


def _delong_test_against_chance(y: pd.Series, yhat: pd.Series):
    """
    Implement the DeLong test to compare the ROC AUC against the 45-degree
    line (AUC = 0.5).

    The DeLong test uses the Central Limit Theorem (CLT) to approximate
    the distribution of the AUC as normal. The test computes the covariance
    matrix of the paired AUC differences and uses it to generate a Z-statistic.
    According to CLT, this Z-statistic will be approximately normally distributed
    for sufficiently large sample sizes (typically n > 30).

    Parameters
    ----------
    None. Relies on the following class attributes:
        - fitted: dictionary of fitted models and metrics
        - target: name of the target variable

    Returns:
    z_stat : Z-statistic
    p_value : p-value of the test
    """
    from scipy.stats import norm

    # Calculate the AUC of the model
    auc = roc_auc_score(y, yhat)

    # Compute the variance of the AUC estimator
    var_auc = _compute_auc_variance(y, yhat)

    # Calculate the Z-statistic against the 45-degree line (AUC=0.5)
    z_stat = np.subtract(auc, 0.5) / np.sqrt(var_auc)

    # Calculate the p-value
    p_value = 2 * (1 - norm.cdf(abs(z_stat)))

    return z_stat, p_value
