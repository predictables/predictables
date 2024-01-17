import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde


def density_by(x: pd.Series, by: pd.Series, ax=None):
    """
    Plot the density of x by the levels of by.

    Parameters
    ----------
    x : pd.Series
        The variable to plot the density of.
    by : pd.Series
        The variable to group by.
    ax : matplotlib.axes.Axes, optional
        The axes to plot on. If None, a new figure and axes is created.

    Returns
    -------
    matplotlib.axes.Axes
        The axes the plot was drawn on.
    """
    density = {}
    for level in by.drop_duplicates().sort_values():
        x = x[by == level]
        density[level] = gaussian_kde(x)

    return density
