import pytest
import polars as pl
from polars.testing import assert_frame_equal
from predictables.core.src.univariate_feature_evaluator.univariate_feature_evaluator import (
    UnivariateFeatureEvaluator,
)
from predictables.core.src.univariate_config import UnivariateConfig
from unittest.mock import MagicMock


@pytest.fixture
def univariate_feature_evaluator():
    train_data = pl.DataFrame(
        {
            "feature1": [1, 2, 3, 4, 5],
            "feature2": [2, 3, 4, 5, 6],
            "target": [0, 1, 0, 1, 0],
            "cv": [1, 2, 1, 2, 1],  # Added cv column
        }
    ).lazy()

    val_data = pl.DataFrame(
        {
            "feature1": [5, 4, 3, 2, 1],
            "feature2": [6, 5, 4, 3, 2],
            "target": [1, 0, 1, 0, 1],
            "cv": [2, 1, 2, 1, 2],  # Added cv column
        }
    ).lazy()

    config = UnivariateConfig(
        model_name="test_model",
        df_train=train_data,
        df_val_=val_data,
        target_column_name="target",
        feature_column_names=["feature1", "feature2"],
        cv_column_name="cv",  # Specified cv column name
    )

    return UnivariateFeatureEvaluator(config=config)


def test_univariate_feature_evaluator_initialization(univariate_feature_evaluator):
    """Test the initialization of the UnivariateFeatureEvaluator."""
    assert (
        univariate_feature_evaluator.config.model_name == "test_model"
    ), f"Expected model_name to be 'test_model', but got {univariate_feature_evaluator.config.model_name}"
    assert (
        univariate_feature_evaluator.df is not None
    ), f"Expected df to be not None, but got {univariate_feature_evaluator.df}"
    assert (
        univariate_feature_evaluator.df_val is not None
    ), f"Expected df_val to be not None, but got {univariate_feature_evaluator.df_val}"
    assert (
        univariate_feature_evaluator.feature_column_names == ["feature1", "feature2"]
    ), f"Expected feature_column_names to be ['feature1', 'feature2'], but got {univariate_feature_evaluator.feature_column_names}"


def test_add_result(univariate_feature_evaluator):
    """Test adding a result to the evaluator."""
    mock_result = pl.DataFrame({"AUC": [0.7]}).lazy()
    univariate_feature_evaluator.add_result(mock_result)

    assert len(univariate_feature_evaluator.results) == 1
    assert_frame_equal(
        univariate_feature_evaluator.results[0].collect(), mock_result.collect()
    )


def test_sort_features_by_metric(univariate_feature_evaluator):
    """Test sorting of features by a specified metric."""
    result1 = pl.DataFrame({"feature": ["feature1"], "AUC": [0.7]}).lazy()
    result2 = pl.DataFrame({"feature": ["feature2"], "AUC": [0.8]}).lazy()

    univariate_feature_evaluator.add_result(result1)
    univariate_feature_evaluator.add_result(result2)

    sorted_features = univariate_feature_evaluator.sort_features_by_metric().collect()

    assert (
        sorted_features["feature"][0] == "feature2"
    ), f"Expected 'feature2' to be first, but got {sorted_features['feature'][0]}"
    assert (
        sorted_features["feature"][1] == "feature1"
    ), f"Expected 'feature1' to be second, but got {sorted_features['feature'][1]}"


def test_sort_features_by_metric_empty_results(univariate_feature_evaluator):
    """Test sorting when there are no results."""
    sorted_features = univariate_feature_evaluator.sort_features_by_metric().collect()
    assert sorted_features.shape == (
        0,
        0,
    ), f"Expected empty DataFrame, but got {sorted_features.shape}"


def test_evaluate_feature(mocker, univariate_feature_evaluator):
    """Test the feature evaluation method."""
    mock_univariate = mocker.patch("predictables.univariate.Univariate", autospec=True)
    mock_results = pl.DataFrame({"AUC": [0.7]}).lazy()
    mock_univariate.return_value.results = mock_results
    mock_univariate.return_value.cv_dict = {
        1: MagicMock(results=mock_results),
        2: MagicMock(results=mock_results),
    }

    univariate_feature_evaluator.evaluate_feature("feature1")

    assert len(univariate_feature_evaluator.results) == 1
    assert_frame_equal(
        univariate_feature_evaluator.results[0].collect(), mock_results.collect()
    )


def test_evaluate_feature_error_handling(mocker, univariate_feature_evaluator):
    """Test error handling in the feature evaluation method."""
    mock_univariate = mocker.patch(
        "predictables.univariate.Univariate", side_effect=Exception("Test error")
    )

    with pytest.raises(Exception, match="Test error"):
        univariate_feature_evaluator.evaluate_feature("feature1")


def test_evaluate_multiple_features(mocker, univariate_feature_evaluator):
    """Test evaluating multiple features."""
    mock_univariate = mocker.patch("predictables.univariate.Univariate", autospec=True)
    mock_results1 = pl.DataFrame({"feature": ["feature1"], "AUC": [0.7]}).lazy()
    mock_results2 = pl.DataFrame({"feature": ["feature2"], "AUC": [0.8]}).lazy()
    mock_univariate.side_effect = [
        MagicMock(
            results=mock_results1,
            cv_dict={
                1: MagicMock(results=mock_results1),
                2: MagicMock(results=mock_results1),
            },
        ),
        MagicMock(
            results=mock_results2,
            cv_dict={
                1: MagicMock(results=mock_results2),
                2: MagicMock(results=mock_results2),
            },
        ),
    ]

    univariate_feature_evaluator.evaluate_feature("feature1")
    univariate_feature_evaluator.evaluate_feature("feature2")

    sorted_features = univariate_feature_evaluator.sort_features_by_metric().collect()

    assert len(univariate_feature_evaluator.results) == 2
    assert (
        sorted_features["feature"][0] == "feature2"
    ), f"Expected 'feature2' to be first, but got {sorted_features['feature'][0]}"
    assert (
        sorted_features["feature"][1] == "feature1"
    ), f"Expected 'feature1' to be second, but got {sorted_features['feature'][1]}"
