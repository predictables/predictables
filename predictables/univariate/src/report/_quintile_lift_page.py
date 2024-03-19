from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from predictables.univariate.src.plots._quintile_lift_plot import quintile_lift_plot
from predictables.util.report import Report


def quintile_lift_page(
    rpt: Report,
    X: pd.Series,  # noqa: N803 -- X is standard machine learning notation
    y: pd.Series,
    cv: pd.Series,
    **kwargs,
) -> Report:
    """Create a page with the quintile lift plot."""
    ax = _get_plot(X, y, cv, **kwargs)
    _save_plot(ax, filename="temp_plot.png")
    return rpt.h2("Univariate Report").h3("Quintile Lift Plot")


def _get_plot(X: pd.Series, y: pd.Series, cv: pd.Series, **kwargs) -> plt.Axes:  # noqa: N803 -- X is standard machine learning notation
    """Get the quintile lift plot."""
    return quintile_lift_plot(X, y, cv, **kwargs)


def _save_plot(ax: plt.Axes | None = None, filename: str = "temp_plot.png") -> None:
    """Save the quintile lift plot to a file."""
    if ax is None:
        raise ValueError("ax must not be None.")
    fig = ax.get_figure()
    fig.savefig(filename)  # type: ignore[union-attr]
