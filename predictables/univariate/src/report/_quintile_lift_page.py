import pandas as pd

from predictables.univariate.src.plots._quintile_lift_plot import quintile_lift_plot
from predictables.util.report import Report


def quintile_lift_page(
    rpt: Report, X: pd.DataFrame, y: pd.Series, cv: pd.Series, **kwargs
) -> Report:
    """
    Create a page with the quintile lift plot.
    """
    ax = _get_plot(X, y, cv, **kwargs)
    _save_plot(ax, filename="temp_plot.png")
    return rpt.h2("Univariate Report").h3("Quintile Lift Plot")


def _get_plot(X, y, cv, **kwargs):
    return quintile_lift_plot(X, y, cv, **kwargs)


def _save_plot(ax, filename="temp_plot.png"):
    fig = ax.get_figure()
    fig.savefig(filename)
