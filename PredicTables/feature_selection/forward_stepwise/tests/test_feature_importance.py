import pandas as pd
import numpy as np

from ..src import feature_importance, fit_model


def test_feature_importance_positive_negative():
    data = pd.DataFrame(
        {
            "feature1": np.random.normal(size=100),
            "feature2": np.random.normal(size=100),
            "target": np.random.normal(size=100) + np.random.normal(size=100),
        }
    )
    features = ["feature1", "feature2"]
    target = "target"
    model = fit_model(data, features, target)
    importance = feature_importance(model, features)
    assert not importance.empty
    assert all(feature in importance["feature"].values for feature in features)


def test_feature_importance_zero_coefficients():
    data = pd.DataFrame(
        {
            "feature1": np.random.normal(size=100),
            "feature2": np.random.normal(size=100),
            "target": np.random.normal(size=100),
        }
    )
    features = ["feature1", "feature2"]
    target = "target"
    model = fit_model(data, features, target)
    importance = feature_importance(model, features)
    assert not importance.empty
    assert all(feature in importance["feature"].values for feature in features)


def test_feature_importance_no_features():
    data = pd.DataFrame(
        {"feature1": np.random.normal(size=100), "target": np.random.normal(size=100)}
    )
    features = []
    target = "target"
    model = fit_model(data, features, target)
    importance = feature_importance(model, features)
    assert importance.empty
