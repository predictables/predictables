from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
import plotly.graph_objects as go

from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import roc_auc_score, roc_curve
from typing import Union, Tuple


# def _calculate_roc_curve(y: pd.Series, yhat:pd.Series, mean_fpr:Union[np.ndarray, None]=None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
#     fpr, tpr, thresholds = roc_curve(y, yhat)
#     if mean_fpr is not None:
#         tpr = np.interp(mean_fpr, fpr, tpr)
#         fpr = mean_fpr
#     return fpr, tpr, thresholds

# def plot_cv_roc_auc( ax: Union[Axes, None] = None, return_all:bool=False, significance_level:float=0.05, figsize:Tuple[int, int]=(10, 6), n_bins:int=10) -> Union[Axes, Tuple[Axes, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, np.ndarray, np.ndarray]]:
#         if ax is None:
#             _, ax = plt.subplots(figsize=figsize)

#         # Plotting the ROC Curve - Mean FPR and TPR

#         # FPR = False Positive Rate, and is the x-axis
#         mean_fpr = np.linspace(0, 1, 100)

#         # TPR = True Positive Rate, and is the y-axis
#         tprs = []

#         # We need to calculate the TPR for each fold at each FPR:
#         # 1. Get the true and predicted values for each fold
#         # 2. Calculate the FPR and TPR for each fold
#         # 3. Interpolate the TPR for each fold at the mean FPR
#         # 4. Calculate the mean TPR across all folds

#         # Loop through each fold
#         for i in range(n_bins):
#             # Get the true and predicted values for the current fold
#             y_true = fitted[i]["val"][target]
#             y_pred = fitted[i]["val"]["prob"]

#             # Calculate the FPR and TPR for the current fold using the
#             # sklearn roc_curve function
#             fpr, tpr, _ = roc_curve(y_true, y_pred)

#             # Interpolate the TPR for the current fold at the mean FPR
#             tprs.append(np.interp(mean_fpr, fpr, tpr))

#             # Plot the ROC curve for the current fold
#             RocCurveDisplay(
#                 fpr=fpr,
#                 tpr=tpr,
#             ).plot(
#                 ax=ax,
#                 # to ensure that the legend is not repeated
#                 # for each fold
#                 # label=None,
#                 alpha=0.4,
#             )

#         # Calculate the mean TPR across all folds
#         mean_tpr = np.mean(tprs, axis=0)

#         # Calculate the mean AUC across all folds, using the trapezoidal rule
#         # to approximate the integral of the ROC curve
#         mean_auc = np.trapz(mean_tpr, mean_fpr)
#         mean_auc = mean_auc
#         sd_auc = np.std(
#             [
#                 roc_auc_score(
#                     fitted[i]["val"][target], fitted[i]["val"]["prob"]
#                 )
#                 for i in range(n_bins)
#             ]
#         )

#         # Calculate the standard deviation of the TPR across all folds
#         std_tpr = np.std(tprs, axis=0)

#         # Our SD band will be mean_tpr +/- 1 std_tpr
#         tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
#         tprs_lower = np.maximum(mean_tpr - std_tpr, 0)

#         # Plotting the ROC Curve - ax.fill_between gives us the SD band
#         ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color="grey", alpha=0.2)

#         # Set the border on the SD band to be darker and thicker
#         ax.spines["bottom"].set_color("grey")
#         ax.spines["left"].set_color("grey")
#         ax.spines["bottom"].set_linewidth(1.5)
#         ax.spines["left"].set_linewidth(1.5)

#         # Plotting the ROC Curve - ax.plot plots the mean ROC curve
#         # ax.plot(mean_fpr, mean_tpr, "b--")
#         ax.plot(mean_fpr, mean_tpr, "b--", label=f"Mean ROC (AUC = {mean_auc:.2f})")

#         # Plotting the random guess line to compare against
#         # ax.plot([0, 1], [0, 1], "k--")
#         ax.plot([0, 1], [0, 1], "k--", label="Random Guess")

#         # Adding a blank bar to the legend to show the SD band
#         # ax.plot([], [], " ")
#         ax.plot([], [], " ", label="Grey band = +/- 1 SD")

#         # Set x and y labels
#         ax.set_xlabel("False Positive Rate")
#         ax.set_ylabel("True Positive Rate")

#         # Perform the DeLong test against the 45-degree line (AUC=0.5)
#         dl_stat, p_value = _delong_test_against_chance()

#         # Annotating the plot with hypothesis testing info
#         significance_message = f"DeLong Test Statistic\nAgainst the\n45-degree Line=\
# \n{dl_stat:.3f}\n\n\
# p-value = {p_value:.2f}"
#         subtitle_message = (
#             f"\nROC AUC is significantly different from 0.5 at the \
# {significance_level:.0%} level"
#             if p_value < significance_level
#             else f"\nROC AUC is not significantly different from 0.5 at the \
# {significance_level:.0%} level"
#         )

#         ax.annotate(
#             significance_message,
#             xy=(0.7, 0.43),
#             xycoords="Axes fraction",
#             fontsize=16,
#             bbox=dict(
#                 boxstyle="round,pad=0.3",
#                 edgecolor="black",
#                 facecolor="aliceblue",
#                 alpha=0.5,
#             ),
#         )

#         ax.set_title(f"ROC Curve (AUC = {mean_auc:.2f}){subtitle_message}")

#         # Add coefficient estimate and significance statement
#         coef, std_err, pvalue = fit.params[1], fit.se[1], fit.pvalues[1]
#         significance_statement = _get_significance_band(pvalue, "coefficient")

#         # 95% confidence interval assuming normal distribution
#         ci_lower = coef - 1.96 * std_err
#         ci_upper = coef + 1.96 * std_err

#         text = f"""Logistic Regression Model Fit
# -----------------------------
# Estimated Coefficient: {coef:.2f}
# 95% Confidence Interval: [{ci_lower:.2f}, {ci_upper:.2f}]
# p-value: {pvalue:.2f}
# {significance_statement}"""

#         # Add annotation inside a box
#         ax.annotate(
#             text,
#             xy=(0.36, 0.19),
#             xycoords="Axes fraction",
#             fontsize=16,
#             bbox=dict(
#                 boxstyle="round,pad=0.3",
#                 edgecolor="black",
#                 facecolor="white",
#                 alpha=0.5,
#             ),
#         )

#         # Create custom legend
#         legend_elements = [
#             Line2D(
#                 [0],
#                 [0],
#                 color="b",
#                 lw=2,
#                 linestyle="--",
#                 label=f"Mean ROC (AUC = {mean_auc:.2f})",
#             ),
#             Line2D([0], [0], color="k", lw=2, linestyle="--", label="Random Guess"),
#             Patch(
#                 facecolor="grey",
#                 edgecolor="grey",
#                 alpha=0.2,
#                 label="Mean(ROC) +/- 1 SD(ROC)",
#             ),
#         ]

#         # ax.legend(loc="lower right", fontsize=16)
#         ax.legend(handles=legend_elements, loc="lower right")

#         ax = rotate_x_lab(ax)
#         ax.figure.tight_layout()
#         if return_all:
#             return (
#                 ax,
#                 mean_fpr,
#                 tprs,
#                 tprs_lower,
#                 tprs_upper,
#                 mean_auc,
#                 mean_tpr,
#                 std_tpr,
#             )
#         else:
#             return ax

# def plotly_cv_roc_auc(y:pd.Series, yhat:pd.Series, significance_level:float=0.05, width:int=800, height:int=800, n_bins:int = 10):
#         mean_fpr = np.linspace(0, 1, 100)
#         tprs = []

#         # Loop through each fold
#         for i in range(n_bins):
#             fpr, tpr, _ = roc_curve(y, yhat)
#             tprs.append(np.interp(mean_fpr, fpr, tpr))

#         # Create empty figure
#         fig = go.Figure()

#         mean_tpr = np.mean(tprs, axis=0)
#         mean_auc = np.trapz(mean_tpr, mean_fpr)

#         std_tpr = np.std(tprs, axis=0)
#         tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
#         tprs_lower = np.maximum(mean_tpr - std_tpr, 0)

#         # Add the shaded region
#         fig.add_trace(
#             go.Scatter(
#                 x=np.concatenate((mean_fpr, mean_fpr[::-1])),
#                 y=np.concatenate((tprs_upper, tprs_lower[::-1])),
#                 fill="toself",
#                 fillcolor="rgba(0,100,80,0.2)",
#                 line=dict(color="rgba(255,255,255,0)"),
#                 name="Mean ROC Curve +/- 1 SD",
#             )
#         )

#         # Loop through each fold
#         for i in range(n_bins):
#             y_true = fitted[i]["val"][target]
#             y_pred = fitted[i]["val"]["prob"]

#             fpr, tpr, _ = roc_curve(y_true, y_pred)
#             tprs.append(np.interp(mean_fpr, fpr, tpr))

#             # Add individselfl ROC lines with hover info, no legend
#             fig.add_trace(
#                 go.Scatter(
#                     x=fpr,
#                     y=tpr,
#                     mode="lines",
#                     hovertemplate=f"Fold {i+1}"
#                     + "<br>FPR: %{x:.3f}\
# <br>TPR: %{y:.3f}<extra></extra>",
#                     legendgroup="Folds",
#                     opacity=0.5,
#                     line=dict(dash="dot"),
#                     showlegend=False,
#                 )
#             )

#         # Add mean ROC line
#         fig.add_trace(
#             go.Scatter(
#                 x=mean_fpr,
#                 y=mean_tpr,
#                 mode="lines+markers",
#                 name=f"Mean ROC (AUC = {mean_auc:.2f})",
#                 hovertemplate="Mean FPR: %{x:.3f}<br>Mean TPR: %{y:.3f}\
# <extra></extra>",
#                 line=dict(color="royalblue", width=1),
#                 marker=dict(color="royalblue", size=5, symbol="circle-open"),
#             )
#         )

#         # Add random guess line
#         fig.add_trace(
#             go.Scatter(
#                 x=[0, 1],
#                 y=[0, 1],
#                 mode="lines",
#                 name="Random Guess",
#                 hovertext="Random Guess",
#                 line=dict(color="black", dash="dash"),
#             )
#         )

#         # Add titles and labels
#         fig.update_layout(
#             title=f"ROC Curve (AUC = {mean_auc:.2f})",
#             xaxis_title="False Positive Rate (FPR) \
# (= 1 - Specificity = FP / (FP + TN))",
#             yaxis_title="True Positive Rate (TPR) (= Sensitivity = TP / (TP + FN))",
#             width=width,
#             height=height,
#             legend=dict(
#                 x=0.65,
#                 y=0.9,
#                 # fontsize=16,
#                 bordercolor="Black",
#                 borderwidth=1,
#             ),
#         )

#         # Add coefficient estimate and significance statement
#         coef, std_err, pvalue = fit.params[1], fit.se[1], fit.pvalues[1]
#         significance_statement = _get_significance_band(pvalue, "coefficient")

#         # 95% confidence interval assuming normal distribution
#         ci_lower = coef - 1.96 * std_err
#         ci_upper = coef + 1.96 * std_err

#         # Add annotation inside a box
#         fig.add_annotation(
#             x=0.75,
#             y=0.25,
#             text=f"Estimated Coefficient: {coef:.2f}<br>\
# 95% Confidence Interval: [{ci_lower:.2f}, {ci_upper:.2f}]<br>\
# p-value: {pvalue:.2f}<br>\
# {significance_statement}",
#             showarrow=False,
#             font=dict(size=14),
#             bgcolor="white",
#             bordercolor="black",
#             borderwidth=1,
#             borderpad=2,
#         )

#         fig.show()


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

    return z_stat[0], p_value[0]


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


def prepare_data_for_plotting(
    y: pd.Series, yhat: pd.Series, n_bins: int
) -> Tuple[pd.Series, pd.DataFrame]:
    mean_fpr = np.linspace(0, 1, 100)
    tprs = []

    for _ in range(n_bins):
        fpr, tpr, _ = roc_curve(y, yhat)
        tprs.append(np.interp(mean_fpr, fpr, tpr))

    mean_fpr = pd.Series(mean_fpr)
    tprs = pd.DataFrame(tprs)

    return mean_fpr, tprs


def plot_individual_roc_curves(
    figax: Union[go.Figure, Axes],
    mean_fpr: pd.Series,
    tprs: pd.DataFrame,
    alpha: float = 0.4,
    backend: str = "matplotlib",
):
    for i, (fpr, tpr) in enumerate(zip(mean_fpr, tprs)):
        if backend == "plotly":
            if isinstance(figax, go.Figure):
                figax.add_trace(
                    go.Scatter(
                        x=fpr,
                        y=tpr,
                        mode="lines",
                        hovertemplate=f"Fold {i+1}"
                        + "<br>FPR: %{x:.3f}\
    <br>TPR: %{y:.3f}<extra></extra>",
                        legendgroup="Folds",
                        opacity=alpha,
                        line=dict(dash="dot"),
                        showlegend=False,
                    )
                )
                return figax
        else:
            if isinstance(figax, Axes):
                RocCurveDisplay(
                    fpr=fpr,
                    tpr=tpr,
                ).plot(ax=figax, alpha=alpha)

                return figax


def calculate_interpolated_mean_tpr(mean_fpr, tprs):
    mean_tpr = np.mean(tprs, axis=0)
    std_tpr = np.std(tprs, axis=0)

    # Interpolate mean_tpr to match the length of mean_fpr
    mean_tpr_interpolated = np.interp(
        mean_fpr, np.linspace(0, 1, len(mean_tpr)), mean_tpr
    )

    return mean_tpr_interpolated, std_tpr


def calculate_mean_std_tpr(tprs):
    mean_tpr = np.mean(tprs, axis=0)
    std_tpr = np.std(tprs, axis=0)
    return mean_tpr, std_tpr


def plot_mean_roc_and_confidence_band_mpl(
    ax: Axes,
    mean_fpr: pd.Series,
    mean_tpr: pd.Series,
    std_tpr: pd.Series,
    mean_auc: float,
    n_sd: int = 1,
    ci_color: str = "grey",
    ci_alpha: float = 0.2,
) -> Axes:
    # Our SD band will be mean_tpr +/- 1 std_tpr
    tprs_upper = np.minimum(mean_tpr + (std_tpr * n_sd), 1)
    tprs_lower = np.maximum(mean_tpr - (std_tpr * n_sd), 0)

    # Plotting the mean ROC curve
    ax.plot(mean_fpr, mean_tpr, "b--", label=f"Mean ROC (AUC = {mean_auc:.2f})")

    # Plotting the confidence band
    ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color=ci_color, alpha=ci_alpha)

    return ax


def configure_annotations_and_legend_mpl(
    ax: Axes, mean_auc: float, std_tpr: pd.Series
) -> Axes:
    """
    Configure annotations and legend for the ROC curve plot.

    Parameters
    ----------
    ax : Axes
        The Axes object to be configured.
    mean_auc : float
        The mean AUC across all folds.
    std_tpr : pd.Series
        The standard deviation of the TPR (true positive rate) across all folds.

    Returns
    -------
    ax : matplotlib.Axes.Axes
        The configured Axes object.
    """
    # Add axis labels and title
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve (AUC = {mean_auc:.2f})")

    # Create custom legend elements
    legend_elements = [
        Line2D(
            [0],
            [0],
            color="b",
            lw=2,
            linestyle="--",
            label=f"Mean ROC (AUC = {mean_auc:.2f})",
        ),
        Patch(
            facecolor="grey",
            edgecolor="grey",
            alpha=0.2,
            label="Mean(ROC) +/- 1 SD(ROC)",
        ),
    ]

    # Add legend to the plot
    ax.legend(handles=legend_elements, loc="lower right")

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


def plot_cv_roc_auc_mpl(
    n_bins: int,
    y: pd.Series,
    yhat: pd.Series,
    alpha: float = 0.05,
    figsize: Tuple[int, int] = (10, 6),
):
    # Data preparation
    mean_fpr, tprs = prepare_data_for_plotting(y, yhat, n_bins)

    # Plot creation
    _, ax = plt.subplots()

    # Plot individual ROC curves
    plot_individual_roc_curves(ax, mean_fpr, tprs)

    # Calculate and plot mean ROC curve and confidence band
    mean_tpr, std_tpr = calculate_interpolated_mean_std_tpr(mean_fpr, tprs)
    mean_auc = auc(mean_fpr, mean_tpr)
    plot_mean_roc_and_confidence_band_mpl(ax, mean_fpr, mean_tpr, std_tpr, mean_auc)

    # Configure annotations and legend
    configure_annotations_and_legend_mpl(ax, mean_auc, std_tpr)

    # Statistical analysis and annotation
    # Placeholder for statistical analysis results
    (
        dl_stat,
        p_value,
    ) = perform_statistical_analysis()  # This function needs to be defined
    add_statistical_analysis_annotation_mpl(ax, dl_stat, p_value, significance_level)

    # Model fit information annotation
    # Placeholder for model fit information
    model_fit_info = get_model_fit_info()  # This function needs to be defined
    add_model_fit_annotation_mpl(ax, model_fit_info)

    # Finalize the plot
    finalize_plot(ax, figsize)

    return ax
