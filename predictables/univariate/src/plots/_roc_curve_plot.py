from __future__ import annotations

import logging
import warnings as warning
from typing import Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go  # type: ignore
from matplotlib.axes import Axes
from scipy.stats import norm  # type: ignore
from sklearn.metrics import roc_auc_score  # type: ignore
from sklearn.metrics import RocCurveDisplay, roc_curve

from predictables.util import DebugLogger, filter_by_cv_fold, get_column_dtype, to_pd_s

dbg = DebugLogger(working_file="_roc_curve_plot.py")


def roc_curve_plot(
    y: pd.Series,
    yhat_proba: pd.Series,
    fold: pd.Series,
    time_series_validation: bool,
    coef: float,
    se: float,
    pvalue: float,
    figsize: Tuple[int, int] = (7, 7),
    n_bins: int = 200,
    cv_alpha: float = 0.4,
    ax: Optional[Axes] = None,
    backend: str = "matplotlib",
) -> Union[go.Figure, Axes]:
    """
    Plot the ROC curve for a single model.

    Parameters
    ----------
    y : pd.Series
        The true labels.
    yhat_proba : pd.Series
        The predicted probabilities.
    fold : pd.Series
        The fold number for each observation.
    time_series_validation : bool
        Whether the data has time series structure. If True, the data will be filtered
        by fold and time, otherwise it will be filtered by fold only.
    coef : float
        The estimated coefficient.
    se : float
        The standard error of the estimated coefficient.
    pvalue : float
        The p-value of the estimated coefficient.
    figsize : Tuple[int, int], optional
        The figure size to use. Defaults to (7, 7).
    n_bins : int, optional
        The number of bins to use when calculating the ROC curve. Generally, the
        more bins, the smoother the curve. Defaults to 200.
    cv_alpha : float, optional
        The opacity of the individual ROC curves.
    ax : matplotlib.axes.Axes, optional
        The Axes object to be configured. If provided, the figure will be plotted
        on the provided Axes object.
    backend : str, optional
        The plotting backend to use. Either "matplotlib" or "plotly". Defaults to
        "matplotlib".

    Returns
    -------
    ax : matplotlib.Axes.Axes
        The configured Axes object.
    """
    if backend == "matplotlib":
        dbg.msg(f"point1: y: {y} | yhat_proba: {yhat_proba} | ROC_0001a")
        return roc_curve_plot_mpl(
            y=y,
            yhat_proba=yhat_proba,
            fold=fold,
            time_series_validation=time_series_validation,
            figsize=figsize if figsize is not None else (7, 7),
            n_bins=n_bins if n_bins is not None else 200,
            cv_alpha=cv_alpha if cv_alpha is not None else 0.4,
            ax=ax,
            coef=coef,
            se=se,
            pvalue=pvalue,
        )
    elif backend == "plotly":
        raise NotImplementedError("Plotly backend not implemented yet.")
    else:
        raise ValueError(
            f"Invalid backend (expecting either 'matplotlib' or 'plotly'): {backend}"
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
    n_bins : int, optional
        The number of bins to use when calculating the ROC curve. Generally, the
        more bins, the smoother the curve. Defaults to 200.

    Returns
    -------
    fpr : pd.Series
        The false positive rate, ranging from 0 to 1, at each threshold.
    tpr : pd.Series
        The true positive rate, ranging from 0 to 1, at each threshold.
    """
    dbg.msg(
        f"y: {y.shape} | yhat_proba: {yhat_proba.shape} | n_bins: {n_bins} | y:\n{y} | yhat_proba:\n{yhat_proba} |\nROC_0001d"
    )
    roc: Tuple[np.ndarray, np.ndarray] = roc_curve(y.round(0).astype(int), yhat_proba)

    # Interpolate the data to get a smoother curve
    fpr: pd.Series = pd.Series(np.linspace(0, 1, n_bins))
    tpr: pd.Series = pd.Series(np.interp(fpr, roc[0], roc[1]))

    return fpr, tpr


def roc_curve_plot_plotly():
    return plt.plot([1, 2, 3], [1, 2, 3])


def plot_individual_roc_curves(
    y: pd.Series,
    yhat_proba: pd.Series,
    curve_name: str = "ROC Curve",
    figax: Optional[Union[go.Figure, Axes]] = None,
    n_bins: int = 200,
    alpha: float = 0.4,
    legendgroup: Optional[str] = None,
    figsize: Tuple[int, int] = (7, 7),
    ax: Optional[Axes] = None,
    fig: Optional[go.Figure] = None,
    backend: str = "matplotlib",
) -> Union[go.Figure, Axes]:
    """
    Plot the ROC curve for each fold.

    Parameters
    ----------
    y : pd.Series
        The true labels.
    yhat_proba : pd.Series
        The predicted probabilities.
    curve_name : str, optional
        The name of the curve to be plotted. If not provided, defaults to "ROC Curve".
    figax : Union[go.Figure, Axes, None], optional
        The plot. Will be ignored if either `fig` (in the case of plotly) or `ax`
        (in the case of matplotlib) is provided.
    n_bins : int, optional
        The number of bins to use when calculating the ROC curve. Generally, the
        more bins, the smoother the curve. Defaults to 200.
    alpha : float, optional
        The opacity of the individual ROC curves.
    legendgroup : Union[str, None], optional
        The legend group to use for the individual ROC curves. If None, no legend
        group is used.
    figsize : Tuple[int, int], optional
        The figure size to use. Defaults to (7, 7).
    ax : matplotlib.axes.Axes, optional
        Alias for figax. If provided and the backend is "matplotlib", figax is ignored.
    fig : plotly.graph_objects.Figure, optional
        Alias for figax if using plotly. If provided and the backend is "plotly",
        figax is ignored.
    plot_title : str, optional
        The title of the plot. Defaults to:
            ROC Curve (AUC = [AUC])
            ROC AUC is [significance_stmnt].
        This will be formatted with the AUC and a statement regarding the
        significance of the AUC estimate.
    backend : str, optional
        The plotting backend to use. Either "matplotlib" or "plotly". Defaults
        to "matplotlib".

    Returns
    -------
    figax : Union[go.Figure, Axes]
        The plot.
    """
    dbg.msg(f"point2: y: {y} | yhat_proba: {yhat_proba} | ROC_0001b")
    fpr, tpr = create_auc_data(y, yhat_proba, n_bins)

    # Handle the case where figax is an alias for either fig or ax
    if (backend == "matplotlib") and (ax is not None):
        figax = ax
        if isinstance(figax, Axes):
            RocCurveDisplay(fpr=fpr, tpr=tpr).plot(
                ax=figax, alpha=alpha, label=curve_name
            )

            return figax
        elif figax is None:
            _, ax = plt.subplots(figsize=figsize)
            RocCurveDisplay(fpr=fpr, tpr=tpr).plot(ax=ax, alpha=alpha, label=curve_name)

            return ax
        else:
            raise TypeError(
                "figax must be a matplotlib.axes.Axes object when using the "
                "matplotlib backend"
            )
    elif (backend == "plotly") and (fig is not None):
        figax = fig
        params = dict(
            x=fpr,
            y=tpr,
            mode="lines",
            hoverdata=pd.DataFrame(dict(fpr=fpr, tpr=tpr)),
            hovertemplate=(
                f"<h4>{curve_name}</h4><br>FPR: {{fpr:.3f}}"
                f"<br>TPR: {{tpr:.3f}}<extra></extra>"
            ),
            opacity=alpha,
            line=dict(dash="dot"),
            showlegend=False,
        )
        if legendgroup is not None:
            params["legendgroup"] = legendgroup

        if isinstance(figax, go.Figure):
            figax.add_trace(go.Scatter(**params))
            return figax
        elif figax is None:
            figax = go.Figure(go.Scatter(**params))
            return figax
        else:
            raise TypeError(
                "figax must be a plotly.graph_objects.Figure object when "
                "using the plotly backend"
            )
    else:
        raise ValueError(
            f"Invalid backend (expecting either 'matplotlib' or 'plotly'): {backend}"
        )


def plot_cv_roc_curves(
    y: pd.Series,
    yhat_proba: pd.Series,
    fold: pd.Series,
    time_series_validation: bool,
    figax: Optional[Union[go.Figure, Axes, None]] = None,
    n_bins: int = 200,
    cv_alpha: float = 0.4,
    ax: Optional[Axes] = None,
    fig: Optional[go.Figure] = None,
    backend: str = "matplotlib",
) -> Union[go.Figure, Axes]:
    """
    Plot the ROC curve for each fold. Filters the data for each fold label,
    then plots the ROC curve for a model trained on that fold.

    Parameters
    ----------
    y : pd.Series
        The true labels.
    yhat_proba : pd.Series
        The predicted probabilities.
    fold : pd.Series
        The fold number for each observation.
    time_series_validation : bool
        Whether the data has time series structure. If True, the data will be filtered
        by fold and time, otherwise it will be filtered by fold only.
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
        Alias for figax if using plotly. If provided and the backend is "plotly",
        figax is ignored.
    backend : str, optional
        The plotting backend to use. Either "matplotlib" or "plotly". Defaults to
        "matplotlib".

    Returns
    -------
    figax : Union[go.Figure, Axes]
        The plot.
    """
    if (backend == "plotly") and (fig is not None):
        figax = fig
    elif (figax is None) and (ax is None):
        _, ax = plt.subplots()
        figax = ax

    for f in fold.drop_duplicates().sort_values().values:
        y_ = filter_by_cv_fold(y, f, fold, time_series_validation, "test")
        yhat_ = filter_by_cv_fold(yhat_proba, f, fold, time_series_validation, "test")
        figax = plot_individual_roc_curves(
            y=y_,
            yhat_proba=yhat_,
            curve_name=f"Fold {f}",
            figax=figax,
            n_bins=n_bins if n_bins is not None else 200,
            alpha=cv_alpha if cv_alpha is not None else 0.4,
            legendgroup="Folds",
            backend=backend if backend is not None else "matplotlib",
        )

    return figax


def calc_auc_curve_data_from_folds(
    y: pd.Series,
    yhat_proba: pd.Series,
    fold: pd.Series,
    time_series_validation: bool,
    n_bins: int = 200,
):
    """
    Calculate the standard error of the ROC curve for each fold. Filters the data for
    each fold label, then plots the ROC curve for a model trained on that fold.

    Parameters
    ----------
    y : pd.Series
        The true labels.
    yhat_proba : pd.Series
        The predicted probabilities.
    fold : pd.Series
        The fold number for each observation.
    time_series_validation : bool
        Whether the data has time series structure. If True, the data will be filtered
        by fold and time, otherwise it will be filtered by fold only.
    n_bins : int, optional
        The number of bins to use when calculating the ROC curve. Generally, the
        more bins, the smoother the curve. Defaults to 200.

    Returns
    -------
    se_fpr : pd.Series
        The standard error of the false positive rate, ranging from 0 to 1, at each
        threshold.
    se_tpr : pd.Series
        The standard error of the true positive rate, ranging from 0 to 1, at each
        threshold.
    """
    # Data preparation

    # Calculate the standard error of the ROC curve for each fold
    fprs, tprs = pd.DataFrame(), pd.DataFrame()

    for f in fold.drop_duplicates().sort_values().values:
        y_ = filter_by_cv_fold(y, f, fold, time_series_validation, "test")
        yhat_ = filter_by_cv_fold(yhat_proba, f, fold, time_series_validation, "test")
        if f > 0:
            fpr, tpr = create_auc_data(y_, yhat_, n_bins)
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
    yhat_proba: pd.Series,
    fold: pd.Series,
    time_series_validation: bool,
    figax: Optional[Union[go.Figure, Axes, None]] = None,
    n_bins: int = 200,
    cv_alpha: float = 0.4,
    ax: Optional[Axes] = None,
    fig: Optional[go.Figure] = None,
    figsize: Tuple[int, int] = (7, 7),
    call_legend: bool = True,
    backend: str = "matplotlib",
) -> Union[go.Figure, Axes]:
    """
    Plot the ROC curve for each fold. Filters the data for each fold label,
    then plots the ROC curve for a model trained on that fold.

    Parameters
    ----------
    y : pd.Series
        The true labels.
    yhat_proba : pd.Series
        The predicted probabilities.
    fold : pd.Series
        The fold number for each observation.
    time_series_validation : bool
        Whether the data has time series structure. If True, the data will be filtered
        by fold and time, otherwise it will be filtered by fold only.
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
        Alias for figax if using plotly. If provided and the backend is "plotly",
        figax is ignored.
    figsize : Tuple[int, int], optional
        The figure size to use. Defaults to (7, 7).
    call_legend : bool, optional
        Whether to call plt.legend() after plotting the ROC curves. Defaults to True.
    backend : str, optional
        The plotting backend to use. Either "matplotlib" or "plotly". Defaults to
        "matplotlib".

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
    elif ax is not None:
        figax = ax
    elif figax is None:
        _, figax = plt.subplots(figsize=figsize)

    dbg.msg(
        f"y: {y.shape} | yhat_proba: {yhat_proba.shape} | fold: {fold.shape} | n_bins: {n_bins} | cv_alpha: {cv_alpha} | ROC_0001c"
    )
    fprs, tprs = calc_auc_curve_data_from_folds(
        y,
        yhat_proba,
        fold,
        time_series_validation,
        n_bins if n_bins is not None else 200,
    )

    if isinstance(figax, Axes):

        def params(col):
            d = dict(alpha=cv_alpha, label="_" + col.replace("_", " ").title(), lw=0.5)
            if col == "mean":
                d["alpha"] = 1
                d["label"] = col.replace("_", " ").title()
                d["lw"] = 2
            elif col.startswith("mean"):
                d["alpha"] = 1
                d["label"] = "_"
                d["lw"] = 0.5
                d["ls"] = "--"
                d["color"] = "grey"
            return d

        for col in fprs.columns:
            x, y = fprs[col], tprs[col]
            figax.plot(x, y, **params(col))

        figax.fill_between(
            fprs["mean"],
            tprs["mean+1"],
            tprs["mean-1"],
            alpha=0.2,
            color="grey",
            label="Mean(ROC) +/- 1 SD(ROC)",
        )

        figax.plot(
            [0, 1], [0, 1], alpha=0.5, ls="--", color="grey", label="Random Guess"
        )

    if call_legend:
        plt.legend()

    return figax


def delong_statistic_annotation_mpl(
    y: pd.Series,
    yhat_proba: pd.Series,
    fold: pd.Series,
    time_series_validation: bool,
    ax: Axes,
):
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
    fold : pd.Series
        The fold number for each observation.
    time_series_validation : bool
        Whether the data has time series structure. If True, the data will be filtered
        by fold and time, otherwise it will be filtered by fold only.
    ax : Axes
        The Axes object to be configured.

    Returns
    -------
    ax : matplotlib.Axes.Axes
        The Axes object annotated with the DeLong test statistic and p-value.
    """
    z, p = _delong_test_against_chance(y, yhat_proba, fold, time_series_validation)

    significance_message = "DeLong Test Statistic\nAgainst the 45-degree Line:\n\n"
    significance_message += f"z = {z:.3f}\n"
    significance_message += f"p-value = {p:.1e}" if p < 1e-3 else f"p-value = {p:.3f}"
    significance_message += "\n" + (
        "\nThe indicated AUC is\nsignificantly different\nfrom random guessing at \n"
        "the 95% confidence level."
        if p < 0.05
        else (
            "The indicated AUC\nis not significantly\ndifferent from random\n"
            "guessing at the 95%\nconfidence level."
        )
    )

    # get the figure size from the Axes object (used to scale the annotation)
    figsize = ax.get_figure().get_size_inches()  # type: ignore

    # add annotation
    ax.annotate(
        significance_message,
        xy=(0.6, 0.2),
        xycoords="axes fraction",
        fontsize=24 * (figsize[0] / 16),
        bbox=dict(
            boxstyle="round,pad=0.3",
            edgecolor="lightgrey",
            facecolor="white",
            alpha=0.8,
        ),
    )

    return ax


def coefficient_annotation_mpl(
    coef: float,
    std_error: float,
    pvalue: float,
    ax: Axes,
    alpha: float = 0.05,
    figsize: Tuple[int, int] = (7, 7),
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
    alpha : float, optional
        The significance level to use. Defaults to 0.05.
    figsize : Tuple[int, int], optional
        The figure size to use. Defaults to (7, 7).

    Returns
    -------
    ax : matplotlib.Axes.Axes
        The Axes object annotated with the model fit information.

    Notes
    -----
    Annotates the plot with the following information:
    - Estimated coefficient
    - Standard error of the estimated coefficient
    - 95% confidence interval (or `1-alpha` confidence interval,
      where `alpha` is the `alpha` parameter)
    - p-value
    - Significance statement

    The significance statement is based on the provided significance level (`alpha`)
    and p-value (`p_value`).
    """
    from scipy.stats import norm

    z_alpha = norm.ppf(1 - alpha / 2)
    ci_lower = coef - z_alpha * std_error
    ci_upper = coef + z_alpha * std_error

    significance_statement = (
        f"Coefficient is significantly\ndifferent from 0 at the {1 - alpha:.1%} level"
        if pvalue < alpha
        else (
            "Coefficient is not significantly\ndifferent from 0 at the "
            f"{1 - alpha:.1%} level"
        )
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
        fontsize=24
        * (figsize[0] / 16),  # scale the font size depending on the figure size
        bbox=dict(
            boxstyle="round,pad=0.3",
            edgecolor="lightgrey",
            facecolor="white",
            alpha=0.8,
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
    return float(roc_auc_score(y, yhat))


def finalize_plot(
    ax: Axes,
    figsize: Tuple[int, int],
    auc: Optional[float] = None,
    auc_p_value: Optional[float] = None,
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
        The area under the ROC curve. Defaults to None. Used to format the plot
        title.
    auc_p_value : float
        The p-value of the area under the ROC curve. Defaults to None. Used to
        format the plot title.

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
    ax.set_title(title, fontsize=30 * (figsize[0] / 16))
    ax.set_xlabel("False Positive Rate", fontsize=24 * (figsize[0] / 16))
    ax.set_ylabel("True Positive Rate", fontsize=24 * (figsize[0] / 16))
    plt.gcf().set_size_inches(figsize)
    plt.tight_layout()

    # show gridlines
    ax.grid(True)
    plt.legend(loc="lower right", fontsize=24 * (figsize[0] / 16))

    return ax


def roc_curve_plot_mpl(
    y: pd.Series,
    yhat_proba: pd.Series,
    fold: pd.Series,
    time_series_validation: bool,
    coef: float,
    se: float,
    pvalue: float,
    figsize: Tuple[int, int] = (7, 7),
    n_bins: int = 200,
    cv_alpha: float = 0.4,
    ax: Optional[Axes] = None,
) -> Axes:
    """
    Plot the ROC curve for each fold. Filters the data for each fold label,
    then plots the ROC curve for a model trained on that fold.

    Parameters
    ----------
    y : pd.Series
        The true labels.
    yhat_proba : pd.Series
        The predicted probabilities.
    fold : pd.Series
        The fold number for each observation.
    time_series_validation : bool
        Whether the data has time series structure. If True, the data will be filtered
        by fold and time, otherwise it will be filtered by fold only.
    coef : float
        The estimated coefficient.
    se : float
        The standard error of the estimated coefficient.
    pvalue : float
        The p-value of the estimated coefficient.
    figsize : Tuple[int, int], optional
        The figure size to use. Defaults to (7, 7).
    n_bins : int, optional
        The number of bins to use when calculating the ROC curve. Generally, the
        more bins, the smoother the curve. Defaults to 200.
    cv_alpha : float, optional
        The opacity of the individual ROC curves.
    ax : matplotlib.axes.Axes, optional
        The Axes object to be configured. If provided, the figure will be plotted
        on the provided Axes object.

    Returns
    -------
    ax : matplotlib.Axes.Axes
        The configured Axes object.
    """
    if isinstance(ax, go.Figure) or (not isinstance(ax, Axes)) and (ax is not None):
        raise TypeError(
            "The ax parameter should be a matplotlib.axes.Axes object when using the matplotlib backend"
        )

    if ax is None:
        _, ax0 = plt.subplots(figsize=figsize)
    else:
        ax0 = ax

    ax0 = plot_roc_auc_curves_and_confidence_bands(
        y,
        yhat_proba,
        fold,
        time_series_validation,
        ax=ax0,
        n_bins=n_bins,
        backend="matplotlib",
        figsize=figsize,
        cv_alpha=cv_alpha,
    )
    ax0 = delong_statistic_annotation_mpl(
        y=y,
        yhat_proba=yhat_proba,
        fold=fold,
        time_series_validation=time_series_validation,
        ax=ax0,
    )
    ax0 = coefficient_annotation_mpl(
        coef=coef,
        std_error=se,
        pvalue=pvalue,
        ax=ax0,
        figsize=figsize,  # type: ignore
    )
    a = auc(y, yhat_proba)
    _, p = _delong_test_against_chance(y, yhat_proba, fold, time_series_validation)
    ax0 = finalize_plot(ax0, figsize=figsize, auc=a, auc_p_value=p)  # type: ignore

    return ax0


def _compute_auc_variance(
    y: pd.Series,
    yhat: pd.Series,
    fold: Optional[pd.Series],
    time_series_validation: bool,
) -> float:
    """
    This function is depricated. Use _empirical_auc_variance instead for new code. I am
    leaving it here for now to avoid breaking existing code, but will comment out the
    implementation and just return the empirical variance so that the tests pass.


    ----------------------------
    ---- vvv Depricated vvv ----
    ----------------------------
    Compute the variance of the AUC estimator. This is used in the
    computation of the DeLong test, and is based on the following paper:

    @article{delong1988comparing,
    title={Comparing the areas under two or more correlated receiver operating
    characteristic curves: a nonparametric approach},
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

    ----------------------------
    ---- ^^^ Depricated ^^^ ----
    ----------------------------

    Returns
    -------
    var_auc : variance of the AUC estimator
    """
    warning.warn(
        "This function is deprecated and will be removed in a future release. "
        "Use `_empirical_auc_variance` instead."
    )
    return _empirical_auc_variance(y, yhat, fold, time_series_validation)  # type: ignore


def _empirical_auc_variance(
    y: pd.Series,
    yhat_proba: pd.Series,
    fold: pd.Series,
    time_series_validation: bool,
    use_bootstrap: bool = False,
    n_bootstraps: int = 1000,
    bagging_fraction: float = 0.8,
) -> float:
    """
    Compute the empirical variance of the AUC estimator. This is used in the
    computation of the DeLong test, and is calculated as the unbiased sample
    variance of the AUC estimator across the folds.

    Parameters
    ----------
    y : pd.Series
        The true labels.
    yhat_proba : pd.Series
        The predicted probabilities.
    fold : pd.Series
        The fold number for each observation.
    time_series_validation : bool
        Whether the data has time series structure. If True, the data will be filtered
        by fold and time, otherwise it will be filtered by fold only.
    use_bootstrap : bool, optional
        Whether to use bootstrapping to compute the variance. Defaults to False.
    n_bootstraps : int, optional
        The number of bootstraps to use. Defaults to 1000.
    bagging_fraction : float, optional
        The fraction of the data to use in each bootstrap. Defaults to 0.8.

    Returns
    -------
    var_auc : float
        The empirical variance of the AUC estimator.
    """
    # Raise an error if there is only one fold
    if (len(fold.unique()) == 1) and (not use_bootstrap):
        raise ValueError(
            "The empirical variance of the AUC estimator cannot be computed with "
            "only one fold. Either pass a different set of fold labels or set "
            "`use_bootstrap` to True."
        )

    if (len(y) == 0) | (len(yhat_proba) == 0) | (len(fold) == 0):
        raise ValueError(
            f"The input arrays are empty:\n y: {len(y)}\n yhat_proba: "
            f"{len(yhat_proba)}\n fold: {len(fold)}"
        )

    if get_column_dtype(y) != "binary":
        raise ValueError(
            f"y must be a binary variable, but it is of type: {get_column_dtype(y)}"
        )

    if to_pd_s(y).isnull().sum() > 0:
        raise ValueError(
            f"y cannot contain missing values, but it contains {y.isnull().sum()} missing values."
        )
    if to_pd_s(yhat_proba).isnull().sum() > 0:
        raise ValueError(
            f"yhat_proba cannot contain missing values, but it contains {yhat_proba.isnull().sum()} missing values."
        )
    if to_pd_s(fold).isnull().sum() > 0:
        raise ValueError(
            f"fold cannot contain missing values, but it contains {fold.isnull().sum()} missing values."
        )

    # If bootstrapping is enabled, compute the variance using bootstrapping
    elif use_bootstrap:
        idx = y.reset_index(drop=True).index.to_series()
        idx_ = np.array(
            [
                idx.sample(frac=bagging_fraction, replace=True)
                for _ in range(n_bootstraps)
            ]
        )
        y_ = np.array([y.iloc[i] for i in idx_])
        yhat_proba_ = np.array([yhat_proba.iloc[i] for i in idx_])
        auc_ = np.array(
            [roc_auc_score(y_[i], yhat_proba_[i]) for i in range(n_bootstraps)]
        )
        return np.var(auc_, ddof=1) * (1 - bagging_fraction) / bagging_fraction

    # Calculate unique folds
    unique_folds = fold.unique()

    # Initialize a list to hold AUC scores for each fold
    auc_scores = []
    counter = 0
    for f in unique_folds:
        try:
            _y_: pd.Series = filter_by_cv_fold(
                y, f, fold, time_series_validation, "test"
            )
            _yhat_proba_: pd.Series = filter_by_cv_fold(
                yhat_proba, f, fold, time_series_validation, "test"
            )
            if len(_y_.drop_duplicates()) < 2:
                auc_scores.append(0)
                continue

            # Calculate AUC score for the current fold
            auc_score = roc_auc_score(_y_, _yhat_proba_)
            auc_scores.append(auc_score)
            counter += 1
        except ValueError as e:
            dbg.msg(
                f"y: {y_} | yhat_proba: {yhat_proba_} | fold: {fold} | time_series_validation: {time_series_validation} | f: {f} | error: {e} | ROC_0001aw"
            )

    # Compute empirical variance of AUC scores
    var_auc = np.var(auc_scores, ddof=1) * (counter - 1) / counter
    return float(var_auc)


def _delong_test_against_chance(
    y: pd.Series, yhat_proba: pd.Series, fold: pd.Series, time_series_validation: bool
) -> Tuple[float, float]:
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
    y : pd.Series
        The true labels.
    yhat_proba : pd.Series
        The predicted probabilities.
    fold : pd.Series
        The fold number for each observation.
    time_series_validation : bool
        Whether the data has time series structure. If True, the data will be filtered
        by fold and time, otherwise it will be filtered by fold only.

    Returns
    -------
    z_stat : float
        The Z-statistic.
    p_value : float
        The p-value.
    """
    if to_pd_s(yhat_proba).nunique() == 1:
        raise ValueError(
            "yhat_proba must have more than one unique value, "
            f"but it has only one unique value: {yhat_proba.unique()}"
        )

    auc = roc_auc_score(y, yhat_proba)
    var_auc = _empirical_auc_variance(y, yhat_proba, fold, time_series_validation)
    if np.sqrt(var_auc) == 0:
        logging.warning(f"Variance of AUC is {var_auc}. Returning p-value of 1.0.")
        return 0.0, 1.0
    z_statistic = np.subtract(auc, 0.5) / np.sqrt(var_auc)
    p_value = 2 * (1 - norm.cdf(abs(z_statistic)))

    return z_statistic, p_value
