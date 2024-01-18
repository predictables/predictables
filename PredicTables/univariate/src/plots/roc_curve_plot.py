from typing import Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from matplotlib.axes import Axes
from sklearn.metrics import RocCurveDisplay, roc_auc_score, roc_curve


def roc_curve_plot(
    y: pd.Series,
    yhat_proba: pd.DataFrame,
    fold: pd.Series,
    figsize: Tuple[int, int] = (8, 8),
    n_bins: int = 200,
    coef: Union[float, None] = None,
    se: Union[float, None] = None,
    pvalue: Union[float, None] = None,
    cv_alpha: float = 0.4,
    ax: Axes = None,
    backend: str = "matplotlib",
) -> Union[go.Figure, Axes]:
    """
    Plot the ROC curve for a single model.

    Parameters
    ----------
    y : pd.Series
        The true labels.
    yhat_proba : pd.DataFrame
        The predicted probabilities.
    fold : pd.Series
        The fold number for each observation.
    figsize : Tuple[int, int], optional
        The figure size to use. Defaults to (8, 8).
    n_bins : int, optional
        The number of bins to use when calculating the ROC curve. Generally, the
        more bins, the smoother the curve. Defaults to 200.
    cv_alpha : float, optional
        The opacity of the individual ROC curves.
    ax : matplotlib.axes.Axes, optional
        The Axes object to be configured. If provided, the figure will be plotted on the provided Axes object.
    backend : str, optional
        The plotting backend to use. Either "matplotlib" or "plotly". Defaults to "matplotlib".

    Returns
    -------
    ax : matplotlib.Axes.Axes
        The configured Axes object.
    """
    if backend == "plotly":
        # return plot_cv_roc_auc_plotly(y, yhat, **kwargs)
        raise NotImplementedError("Plotly backend not implemented yet.")
    else:
        return roc_curve_plot_mpl(
            y=y,
            yhat_proba=yhat_proba,
            fold=fold,
            figsize=figsize,
            n_bins=n_bins,
            coef=coef,
            se=se,
            pvalue=pvalue,
            cv_alpha=cv_alpha,
            ax=ax,
        )


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
    ax: Union[Axes, None] = None,
    fig: Union[go.Figure, None] = None,
    figsize: Tuple[int, int] = (8, 8),
    backend: str = "matplotlib",
) -> Union[go.Figure, Axes, None]:
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
    figsize : Tuple[int, int], optional
        The figure size to use. Defaults to (8, 8).
    backend : str, optional
        The plotting backend to use. Either "matplotlib" or "plotly". Defaults to "matplotlib".

    Returns
    -------
    figax : Union[go.Figure, Axes]
        The plot.

    Raises
    ------
    TypeError
        If ax is not a matplotlib.axes.Axes object when using the matplotlib backend.
    NotImplementedError
        If using the plotly backend, which is not yet implemented.
    """

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

        if isinstance(figax, Axes):
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


def delong_statistic_annotation_mpl(y: pd.Series, yhat_proba: pd.Series, ax: Axes):
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

    significance_message = "DeLong Test Statistic\nAgainst the 45-degree Line:\n\n"
    significance_message += f"z = {z:.3f}\n"
    significance_message += f"p-value = {p:.1e}" if p < 1e-3 else f"p-value = {p:.3f}"
    significance_message += "\n" + (
        "\nThe indicated AUC is\nsignificantly different\nfrom random guessing at \nthe 95% confidence level."
        if p < 0.05
        else "The indicated AUC is not significantly different from random guessing at the 95% confidence level."
    )

    # add annotation
    ax.annotate(
        significance_message,
        xy=(0.6, 0.2),
        xycoords="axes fraction",
        fontsize=12,
        bbox=dict(
            boxstyle="round,pad=0.3", edgecolor="black", facecolor="white", alpha=0.8
        ),
    )

    return ax


def coefficient_annotation_mpl(
    coef: float,
    std_error: float,
    pvalue: float,
    ax: Axes,
    alpha: float = 0.05,
):
    """
    Annotate the plot with logistic regression model fit information.

    Parameters
    ----------
    coef : float
        The estimated coefficient.
    std_error : float
        The standard error of the estimated coefficient.
    pvalue : float
        The p-value of the estimated coefficient.
    ax : Axes
        The Axes object to be configured.
    alpha : float
        The significance level to use. Defaults to 0.05.

    Returns
    -------
    ax : matplotlib.Axes.Axes
        The Axes object annotated with the model fit information.

    Notes
    -----
    Annotates the plot with the following information:
    - Estimated coefficient
    - Standard error of the estimated coefficient
    - 95% confidence interval (or `1-alpha` confidence interval, where `alpha` is the `alpha` parameter)
    - p-value
    - Significance statement

    The significance statement is based on the provided significance level (`alpha`) and p-value (`p_value`).
    """
    from scipy.stats import norm

    z_alpha = norm.ppf(1 - alpha / 2)
    ci_lower = coef - z_alpha * std_error
    ci_upper = coef + z_alpha * std_error

    significance_statement = (
        f"Coefficient is significantly\ndifferent from 0 at the {1 - alpha:.1%} level"
        if pvalue < alpha
        else f"Coefficient is not significantly\ndifferent from 0 at the {1 - alpha:.1%} level"
    )

    annotation_text = "Logistic Regression Fit Statistics\n======================\n\n"
    annotation_text += (
        f"Estimated Coefficient: {coef:.1e}\n"
        if coef < 1e-3
        else f"Estimated Coefficient: {coef:.2f}\n"
    )
    annotation_text += (
        f"SE(Coefficient): {std_error:.1e}\n"
        if std_error < 1e-3
        else f"SE(Coefficient): {std_error:.2f}\n"
    )
    annotation_text += (
        f"95% Confidence Interval:\n[{ci_lower:.1e}, {ci_upper:.1e}]\n"
        if ci_lower < 1e-3
        else f"95% Confidence Interval:\n[{ci_lower:.2f}, {ci_upper:.2f}]\n"
    )
    annotation_text += (
        f"p-value: {pvalue:.1e}\n\n" if pvalue < 1e-3 else f"p-value: {pvalue:.2f}\n\n"
    )
    annotation_text += f"{significance_statement}"

    ax.annotate(
        annotation_text,
        xy=(0.1, 0.05),
        xycoords="axes fraction",
        fontsize=12,
        bbox=dict(
            boxstyle="round,pad=0.3", edgecolor="black", facecolor="white", alpha=0.8
        ),
    )

    return ax


def auc(y: pd.Series, yhat: pd.Series) -> float:
    """
    Compute the area under the ROC curve. Uses the stock implementation from
    scikit-learn.

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

    return float(roc_auc_score(y, yhat))


def finalize_plot(
    ax: Axes,
    figsize: Tuple[int, int],
    auc: Union[float, None] = None,
    auc_p_value: Union[float, None] = None,
) -> Axes:
    """
    Finalize the plot by adjusting the figure size and layout.

    Parameters
    ----------
    ax : Axes
        The Axes object to be configured.
    figsize : Tuple[int, int]
        The figure size to use.
    auc : float
        The area under the ROC curve. Defaults to None. Used to format the plot title.
    auc_p_value : float
        The p-value of the area under the ROC curve. Defaults to None. Used to format the plot title.

    Returns
    -------
    ax : matplotlib.Axes.Axes
        The configured Axes object.
    """
    if auc_p_value is None:
        auc_p_value = 1000.0
    if auc is None:
        auc = 1000.0

    title = f"ROC Curve (AUC = {auc:.1%})\n"
    title += (
        "ROC AUC is significantly different from 0.5 at the 95% level."
        if auc_p_value < 0.05
        else "ROC AUC is not significantly different from 50% at the 95% level."
    )
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    plt.gcf().set_size_inches(figsize)
    plt.tight_layout()

    # show gridlines
    ax.grid(True)
    plt.legend(loc="lower right", fontsize=12)

    return ax


def roc_curve_plot_mpl(
    y: pd.Series,
    yhat_proba: pd.Series,
    fold: pd.Series,
    figsize: Tuple[int, int] = (8, 8),
    n_bins: int = 200,
    coef: Union[float, None] = None,
    se: Union[float, None] = None,
    pvalue: Union[float, None] = None,
    cv_alpha: float = 0.4,
    ax: Union[Axes, None] = None,
) -> Union[Axes, None]:
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
    figsize : Tuple[int, int], optional
        The figure size to use. Defaults to (8, 8).
    n_bins : int, optional
        The number of bins to use when calculating the ROC curve. Generally, the
        more bins, the smoother the curve. Defaults to 200.
    cv_alpha : float, optional
        The opacity of the individual ROC curves.
    ax : matplotlib.axes.Axes, optional
        The Axes object to be configured. If provided, the figure will be plotted on the provided Axes object.

    Returns
    -------
    ax : matplotlib.Axes.Axes
        The configured Axes object.
    """
    if isinstance(ax, go.Figure):
        raise TypeError(
            "ax must be a matplotlib.axes.Axes object when using the matplotlib backend"
        )
    if ax is not None:
        pass
    else:
        _, ax = plt.subplots(figsize=figsize)

        ax = plot_roc_auc_curves_and_confidence_bands(
            y,
            yhat_proba,
            fold,
            ax=(ax if not isinstance(ax, go.Figure) else None),
            n_bins=n_bins,
            backend="matplotlib",
            figsize=figsize,
            cv_alpha=cv_alpha,
        )
        ax = delong_statistic_annotation_mpl(y=y, yhat_proba=yhat_proba, ax=ax)
        ax = coefficient_annotation_mpl(coef=coef, std_error=se, pvalue=pvalue, ax=ax)
        a = auc(y, yhat_proba)
        _, p = _delong_test_against_chance(y, yhat_proba)
        ax = finalize_plot(ax, figsize=figsize, auc=a, auc_p_value=p)
        return ax


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