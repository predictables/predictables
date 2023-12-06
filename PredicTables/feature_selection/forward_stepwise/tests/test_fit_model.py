import pytest
import pandas as pd
import numpy as np
from ..src import fit_model


def test_fit_model_single_feature():
    data = pd.DataFrame(
        {"feature1": np.random.normal(size=100), "target": np.random.normal(size=100)}
    )
    features = ["feature1"]
    target = "target"
    model = fit_model(data, features, target)
    assert model is not None
    assert "feature1" in model.model.exog_names


def test_fit_model_multiple_features():
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
    assert model is not None
    assert all(feature in model.model.exog_names for feature in features)


def test_fit_model_invalid_feature():
    data = pd.DataFrame(
        {"feature1": np.random.normal(size=100), "target": np.random.normal(size=100)}
    )
    features = ["feature1", "invalid_feature"]
    target = "target"
    with pytest.raises(Exception):
        fit_model(data, features, target)
