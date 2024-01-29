from typing import Any

def partial_dependence(
    estimator,
    X,
    features,
    *,
    sample_weight: Any | None = ...,
    categorical_features: Any | None = ...,
    feature_names: Any | None = ...,
    response_method: str = ...,
    percentiles=...,
    grid_resolution: int = ...,
    method: str = ...,
    kind: str = ...
): ...
