import pandas as pd

from bokeh.plotting import figure

from predictables.app.src.pca import pca as _pca


def biplot(X: pd.DataFrame) -> figure:
    """Create a biplot for the given data."""
    pca = _pca(X)

    scores = pca.transform(pca.components_)

    p = figure(
        x_axis_label="Principal Component 1",
        y_axis_label="Principal Component 2",
        width=1000,
        height=600,
    )

    p.circle(x=scores[:, 0], y=scores[:, 1], size=8, fill_color="blue", fill_alpha=0.5, line_color="black", line_width=1)

    return p