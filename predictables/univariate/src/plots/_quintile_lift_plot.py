from __future__ import annotations


import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objs as go  # type: ignore[import-untyped]
import polars as pl
from scipy.stats import entropy as kl_divergence  # type: ignore[import-untyped]

from predictables.util import to_pd_s
from predictables.util.stats import gini_coefficient

from .util import rotate_x_lab


def quintile_lift_plot(
    feature: pd.Series | pl.Series,
    observed_target: pd.Series | pl.Series,
    modeled_target: pd.Series | pl.Series,
    ax: plt.Axes | None = None,
    backend: str = "matplotlib",
    figsize: tuple[float, float] | None = None,
    **kwargs,
) -> plt.Axes | go.Figure:
    """Plot the quintile lift for a given feature and target.

    The function calculates the mean observed target and modeled target for each
    quintile of the modeled target, and plots them as a bar chart. The function
    also calculates the KL divergence and Gini coefficient between the observed
    and modeled targets, and adds them as annotations to the plot.

    Parameters
    ----------
    feature : pd.Series | pl.Series,
        A Pandas Series containing the feature data.
    observed_target : pd.Series | pl.Series,
        A Pandas Series containing the observed target data.
    modeled_target : pd.Series | pl.Series,
        A Pandas Series containing the modeled target data.
    ax: Axes, optional
        The Matplotlib axis object to plot the quintile lift on. If not provided,
        a new figure and axis object will be created.
    backend : str, optional
        The plotting backend to use. Default is 'matplotlib'.
    figsize : tuple[int, int], optional
        The figure size. Default is (7, 7).
    **kwargs
        Additional keyword arguments to pass to the plotting function.

    Returns
    -------
    matplotlib.axes.Axes or plotly.graph_objs._figure.Figure
        The plot.
    """
    if backend not in ["matplotlib", "plotly"]:
        raise ValueError(f"Unknown backend: {backend}")

    # Use the figsize parameter, then kwarg, then default to (7, 7)
    figsize0 = figsize if figsize is not None else kwargs.get("figsize", (7, 7))

    if ax is None:
        _, ax = plt.subplots(figsize=figsize0)

    params = dict(
        feature=feature,
        observed_target=observed_target,
        modeled_target=modeled_target,
        figsize=figsize0,
        **kwargs,
    )

    return (
        quintile_lift_plot_matplotlib(**params, ax=ax)
        if backend == "matplotlib"
        else quintile_lift_plot_plotly(**params)
    )


def quintile_lift_plot_matplotlib(
    feature: pd.Series | pl.Series,
    observed_target: pd.Series | pl.Series,
    modeled_target: pd.Series | pl.Series,
    ax: plt.Axes | None = None,
    figsize: tuple[int, int] = (7, 7),
) -> plt.Axes:
    """Plot the quintile lift for a given feature and target.

    The function calculates the mean observed target and modeled target for each
    quintile of the modeled target, and plots them as a bar chart. The function
    also calculates the KL divergence and Gini coefficient between the observed
    and modeled targets, and adds them as annotations to the plot.

    Parameters
    ----------
    feature : pd.Series | pl.Series,
        A Pandas Series containing the feature data.
    observed_target : pd.Series | pl.Series,
        A Pandas Series containing the observed target data.
    modeled_target : pd.Series | pl.Series,
        A Pandas Series containing the modeled target data.
    ax : matplotlib.axes.Axes, optional
        The Matplotlib axis object to plot the quintile lift on. If not provided,
        a new figure and axis object will be created.
    modeled_color : str, optional
        The color of the modeled target bars. Default is 'red'.
    observed_color : str, optional
        The color of the observed target bars. Default is 'lightgreen'.
    edge_color : str, optional
        The color of the bar edges. Default is 'black'.
    alpha : float, optional
        The transparency of the bars. Default is 0.5.
    figsize : tuple[int, int], optional
        The figure size. Default is (7, 7).

    Returns
    -------
    matplotlib.axes.Axes
        The Matplotlib axis object with the quintile lift plot.
    """
    lift_df = _prep_data(feature, observed_target, modeled_target)

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    font_scale_fct = 1.25 * figsize[0] / 16 if figsize is not None else 1

    bars1, bars2 = _make_bars(lift_df, ax)

    # Add data labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height:.2f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, -3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=20 * font_scale_fct,
                bbox={
                    "boxstyle": "round,pad=0.2",
                    "edgecolor": "black",
                    "facecolor": "white",
                    "alpha": 0.9,
                },
            )

    ax.set_xticks(lift_df["quintile"])
    ax.set_xlabel("Modeled Quintile", fontsize=20 * font_scale_fct)
    ax.set_ylabel("Mean Target", fontsize=20 * font_scale_fct)
    ax.legend(fontsize=20 * font_scale_fct)

    # KL Divergence calculation
    kl_div = _kl_divergence(lift_df)
    gini_coeff = gini_coefficient(observed_target.to_list(), modeled_target.to_list())

    # Add KL divergence and Gini coefficient as annotation to the plot
    ax.annotate(
        f"KL Divergence: {kl_div:.3f}\nGini Coefficient: {gini_coeff:.3f}",
        xy=(0.75, 0.05),
        xycoords="axes fraction",
        fontsize=20 * font_scale_fct,
        ha="center",
        bbox={
            "boxstyle": "round,pad=0.25",
            "edgecolor": "black",
            "facecolor": "white",
            "alpha": 0.85,
        },
    )

    ax.set_title("Qunitile Lift Plot")
    ax = rotate_x_lab(ax)

    # show gridlines
    ax.grid(True)
    return ax


def quintile_lift_plot_plotly(
    feature: pd.Series | pl.Series,
    observed_target: pd.Series | pl.Series,
    modeled_target: pd.Series | pl.Series,
) -> go.Figure:
    """Plot the quintile lift for a given feature and target.

    The function calculates the mean observed target and modeled target for each
    quintile of the modeled target, and plots them as a bar chart. The function
    also calculates the KL divergence and Gini coefficient between the observed
    and modeled targets, and adds them as annotations to the plot.

    Parameters
    ----------
    feature : pd.Series | pl.Series,
        A Pandas Series containing the feature data.
    observed_target : pd.Series | pl.Series,
        A Pandas Series containing the observed target data.
    modeled_target : pd.Series | pl.Series,
        A Pandas Series containing the modeled target data.

    Returns
    -------
    plotly.graph_objs._figure.Figure
        The plot.
    """
    lift_df = _prep_data(feature, observed_target, modeled_target)

    ax = plt.gca()  # Create a dummy axis to pass to _make_bars

    bars1, bars2 = _make_bars(lift_df, ax, backend="plotly")

    fig = go.Figure(data=[bars1, bars2])

    # Add data labels
    for bars in [bars1, bars2]:
        for bar in bars:
            fig.add_annotation(
                x=bar.x,
                y=bar.y,
                text=f"{bar.y:.2f}",
                showarrow=False,
                font={"size": 16},
            )

    fig.update_layout(
        xaxis_title="Modeled Quintile",
        yaxis_title="Mean Target",
        title="Quintile Lift Plot",
        legend={"font": {"size": 16}},
    )

    # KL Divergence calculation
    kl_div = _kl_divergence(lift_df)
    gini_coeff = gini_coefficient(observed_target.to_list(), modeled_target.to_list())

    # Add KL divergence and Gini coefficient as annotation to the plot
    fig.add_annotation(
        x=0.75,
        y=0.05,
        text=f"KL Divergence: {kl_div:.3f}<br>Gini Coefficient: {gini_coeff:.3f}",
        showarrow=False,
        font={"size": 16},
    )

    return fig


def _prep_data(
    feature: pd.Series | pl.Series,
    observed_target: pd.Series | pl.Series,
    modeled_target: pd.Series | pl.Series,
) -> pd.DataFrame:
    """Prepare the data for the quintile lift plot.

    Parameters
    ----------
    feature : pd.Series | pl.Series,
        A Pandas Series containing the feature data.
    observed_target : pd.Series | pl.Series,
        A Pandas Series containing the observed target data.
    modeled_target : pd.Series | pl.Series,
        A Pandas Series containing the modeled target data.

    Returns
    -------
    pd.DataFrame
        A Pandas DataFrame containing the data for the quintile lift plot.
    """
    # Create DataFrame to hold all the data
    df = pd.DataFrame(
        {
            "feature": to_pd_s(feature),
            "observed_target": to_pd_s(observed_target),
            "modeled_target": to_pd_s(modeled_target),
        }
    )
    df["quintile"] = _make_quintiles(df["modeled_target"])

    # Calculate the mean target and modeled target for each quintile
    return (
        df.groupby("quintile")
        .agg(
            observed_target_mean=("observed_target", "mean"),
            modeled_target_mean=("modeled_target", "mean"),
        )
        .reset_index()
    )


def _make_quintiles(modeled_target: pd.Series | pl.Series) -> pd.Series:
    """Create quintile bins based on the modeled target.

    If there are n < 5 unique values, don't bin quintiles --
    instead, bin into n bins based on the modeled target.

    Parameters
    ----------
    modeled_target : pd.Series | pl.Series,
        A Pandas Series containing the modeled target data.

    Returns
    -------
    pd.Series
        A Pandas Series containing the quintile bins.
    """
    # Validate inputs
    if not isinstance(modeled_target, pd.Series) and not isinstance(
        modeled_target, pl.Series
    ):
        raise TypeError(
            "Expected modeled_target to be a Pandas or Polars Series, "
            f"got {type(modeled_target)}"
        )

    modeled_target = to_pd_s(modeled_target)

    # Return quintile bins
    return (
        (
            pd.qcut(
                modeled_target,
                len(modeled_target.unique()),
                labels=False,
                duplicates="drop",
            )
            + 1
        )
        if len(modeled_target.unique()) < 5  # (5 is based off of quintiles)
        else pd.qcut(modeled_target, 5, labels=False, duplicates="drop") + 1
    )


def _make_bars(
    df: pd.DataFrame,
    ax: plt.Axes,
    backend: str = "matplotlib",
    modeled_color: str = "red",
    observed_color: str = "lightgreen",
    edge_color: str = "black",
    alpha: float = 0.5,
) -> tuple[plt.Bar, plt.Bar] | tuple[go.Bar, go.Bar]:
    if backend == "matplotlib":
        bars1 = ax.bar(
            df["quintile"] - 0.2,
            df["observed_target_mean"],
            0.4,
            label="Observed",
            color=observed_color,
            edgecolor=edge_color,
            alpha=alpha,
        )
        bars2 = ax.bar(
            df["quintile"] + 0.2,
            df["modeled_target_mean"],
            0.4,
            label="Modeled",
            color=modeled_color,
            edgecolor=edge_color,
            alpha=alpha,
        )
    elif backend == "plotly":
        bars1 = go.Bar(
            x=df["quintile"],
            y=df["observed_target_mean"],
            name="Observed",
            marker_color=observed_color,
        )
        bars2 = go.Bar(
            x=df["quintile"],
            y=df["modeled_target_mean"],
            name="Modeled",
            marker_color=modeled_color,
        )
    else:
        raise ValueError(f"Unknown backend: {backend}")

    return bars1, bars2


def _kl_divergence(df: pd.DataFrame) -> float:
    return kl_divergence(
        df["observed_target_mean"].values, df["modeled_target_mean"].values
    )
