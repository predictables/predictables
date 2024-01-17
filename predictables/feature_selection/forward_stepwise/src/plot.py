from typing import List

import pandas as pd
import plotly.graph_objects as go
import statsmodels.api as sm
from plotly.subplots import make_subplots


def plot(
    importance_df: pd.DataFrame,
    model: sm.regression.linear_model.RegressionResultsWrapper,
    model_history: List[str],
    data: pd.DataFrame,
    features: List[str],
    target: str,
) -> None:
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Feature Importance",
            "Residuals Plot",
            "Model Fit Plot",
            "Coefficient Path Plot",
        ),
    )

    fig.add_trace(
        go.Bar(
            x=importance_df["feature"],
            y=importance_df["coefficient"],
            name="Feature Importance",
        ),
        row=1,
        col=1,
    )

    predictions = model.predict(data[features])
    residuals = data[target] - predictions
    fig.add_trace(
        go.Scatter(x=predictions, y=residuals, mode="markers", name="Residuals"),
        row=1,
        col=2,
    )

    fig.add_trace(
        go.Scatter(x=data[target], y=predictions, mode="markers", name="Model Fit"),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Line(x=data[target], y=data[target], showlegend=False), row=2, col=1
    )

    coefficients = {feature: [] for feature, _ in model_history}
    for feature, mdl in model_history:
        for f in coefficients:
            coefficients[f].append(mdl.params.get(f, 0))

    for feature, values in coefficients.items():
        fig.add_trace(
            go.Scatter(
                x=list(range(1, len(model_history) + 1)),
                y=values,
                mode="lines+markers",
                name=feature,
            ),
            row=2,
            col=2,
        )

    fig.update_layout(
        title_text="Feature Importance Based on Forward Stepwise Selection",
        height=800,
        width=800,
    )
    fig.show()
