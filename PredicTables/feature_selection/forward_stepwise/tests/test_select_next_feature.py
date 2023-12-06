import pandas as pd
import numpy as np
from ..src import select_next_feature


def test_select_next_feature_with_strong_relationship():
    data = pd.DataFrame(
        {
            "feature1": np.random.normal(size=100),
            "feature2": np.random.normal(size=100),
            "feature3": np.random.normal(size=100),
            "target": np.random.normal(size=100),
        }
    )
    data["target"] += 3 * data["feature1"]  # Strong relationship with feature1
    selected_features = []
    remaining_features = ["feature1", "feature2", "feature3"]
    target = "target"
    assert (
        select_next_feature(data, selected_features, remaining_features, target)
        == "feature1"
    )


def test_select_next_feature_no_significant_features():
    data = pd.DataFrame(
        {
            "feature1": np.random.normal(size=100),
            "feature2": np.random.normal(size=100),
            "target": np.random.normal(size=100),
        }
    )
    selected_features = []
    remaining_features = ["feature1", "feature2"]
    target = "target"
    assert (
        select_next_feature(data, selected_features, remaining_features, target) is None
    )


def test_select_next_feature_empty_remaining_features():
    data = pd.DataFrame(
        {"feature1": np.random.normal(size=100), "target": np.random.normal(size=100)}
    )
    selected_features = ["feature1"]
    remaining_features = []
    target = "target"
    assert (
        select_next_feature(data, selected_features, remaining_features, target) is None
    )
