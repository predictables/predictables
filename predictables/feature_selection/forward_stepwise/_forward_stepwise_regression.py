from typing import List

import pandas as pd

from .src import feature_importance, fit_model, select_next_feature


def forward_stepwise_regression(data: pd.DataFrame, target: str) -> List[str]:
    selected_features = []
    remaining_features = list(data.columns)
    remaining_features.remove(target)
    model_history = []

    while remaining_features:
        next_feature = select_next_feature(
            data, selected_features, remaining_features, target
        )
        if next_feature:
            selected_features.append(next_feature)
            remaining_features.remove(next_feature)
            model = fit_model(data, selected_features, target)
            model_history.append((next_feature, model))
        else:
            break

    importance = feature_importance(model, selected_features)
    return selected_features, model, model_history, importance
