import pytest
import pandas as pd
import polars as pl
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from predictables.feature_selection.src._backward_stepwise import (
    initialize_feature_set,
    calculate_all_feature_correlations,
    identify_highly_correlated_pairs,
    generate_X_y,
    evaluate_feature_removal_impact,
    select_feature_to_remove,
    backward_stepwise_feature_selection,
)


@pytest.fixture
def synthetic_data():
    data_size = 100
    # Create features with varying degrees of correlation
    features = {
        "X1": np.random.default_rng(seed=42).random(data_size),
        "X2": np.random.default_rng(seed=42).random(data_size) * 0.5
        + np.random.default_rng(seed=42).random(data_size) * 0.5,
        "X3": np.random.default_rng(seed=42).random(data_size) * 0.1
        + np.random.default_rng(seed=42).random(data_size) * 0.9,
    }

    df = pd.DataFrame(features)

    # X4 is highly correlated with X1 (it is a linear combination of X1 and some noise)
    df["X4"] = 0.9 * df["X1"] + 0.1 * np.random.default_rng(seed=42).random(data_size)

    # X5 is not very correlated with others
    df["X5"] = np.random.default_rng(seed=42).random(data_size)

    y = np.random.default_rng(seed=42).integers(0, 2, size=data_size)
    return df, y


@pytest.fixture
def synthetic_data_pl_df(synthetic_data):
    X, y = synthetic_data
    return pl.DataFrame(X), y


@pytest.fixture
def synthetic_data_pl_lazy(synthetic_data_pl_df):
    X, y = synthetic_data_pl_df
    return X.lazy(), y


def test_initialize_feature_set(
    synthetic_data, synthetic_data_pl_df, synthetic_data_pl_lazy
):
    X, _ = synthetic_data
    features = initialize_feature_set(X)
    expected_features = ["X1", "X2", "X3", "X4", "X5"]
    assert isinstance(
        features, list
    ), f"The output should be a list, but got {type(features)}."
    assert (
        set(features) == set(expected_features)
    ), f"The list of features should match the columns of the DataFrame, but got\n{features}\ninstead of\n{expected_features}."
    assert (
        len(features) == len(X.columns)
    ), f"The number of features should match the number of columns in the DataFrame, but got\n{len(features)}\ninstead of\n{len(X.columns)}."
    assert (
        features == list(X.columns)
    ), f"The order of features should match the order of columns in the DataFrame, but got\n{features}\ninstead of\n{list(X.columns)}."

    X_pl, _ = synthetic_data_pl_df
    features_pl = initialize_feature_set(X_pl)
    assert isinstance(
        features_pl, list
    ), f"The output should be a list, but got {type(features_pl)}."
    assert (
        set(features_pl) == set(expected_features)
    ), f"The list of features should match the columns of the DataFrame, but got\n{features_pl}\ninstead of\n{expected_features}."
    assert (
        len(features_pl) == len(X.columns)
    ), f"The number of features should match the number of columns in the DataFrame, but got\n{len(features_pl)}\ninstead of\n{len(X.columns)}."
    assert (
        features_pl == list(X.columns)
    ), f"The order of features should match the order of columns in the DataFrame, but got\n{features_pl}\ninstead of\n{list(X.columns)}."

    X_lazy, _ = synthetic_data_pl_lazy
    features_lazy = initialize_feature_set(X_lazy)
    assert isinstance(
        features_lazy, list
    ), f"The output should be a list, but got {type(features_lazy)}."
    assert (
        set(features_lazy) == set(expected_features)
    ), f"The list of features should match the columns of the DataFrame, but got\n{features_lazy}\ninstead of\n{expected_features}."
    assert (
        len(features_lazy) == len(X.columns)
    ), f"The number of features should match the number of columns in the DataFrame, but got\n{len(features_lazy)}\ninstead of\n{len(X.columns)}."
    assert (
        features_lazy == list(X.columns)
    ), f"The order of features should match the order of columns in the DataFrame, but got\n{features_lazy}\ninstead of\n{list(X.columns)}."
    assert (
        features_lazy == features_pl
    ), f"The output of the function should be the same for both pandas and polars DataFrames, but got\n{features_lazy}\nand\n{features_pl}."
    assert (
        features_lazy == features
    ), f"The output of the function should be the same for both pandas and polars DataFrames, but got\n{features_lazy}\nand\n{features}."
    assert (
        features_pl == features
    ), f"The output of the function should be the same for both pandas and polars DataFrames, but got\n{features_pl}\nand\n{features}."
    assert (
        features_lazy is not features
    ), "The output of the function should be a new list, but got the same object for both pandas and polars DataFrames."
    assert (
        features_lazy is not features_pl
    ), "The output of the function should be a new list, but got the same object for both pandas and polars DataFrames."
    assert (
        features_pl is not features
    ), "The output of the function should be a new list, but got the same object for both pandas and polars DataFrames."


def test_calculate_all_feature_correlations(
    synthetic_data, synthetic_data_pl_df, synthetic_data_pl_lazy
):
    X, _ = synthetic_data
    correlations = calculate_all_feature_correlations(X)
    assert isinstance(
        correlations, pd.DataFrame
    ), f"The output should be a pandas DataFrame, but got {type(correlations)}."
    assert (
        correlations.shape == (len(X.columns), len(X.columns))
    ), f"The correlation matrix should be square and match the number of features, but:\nNumber of features: {len(X.columns)}\nShape of the correlation matrix: {correlations.shape}."
    assert (
        correlations.index == X.columns
    ).all(), f"The row indices should match the feature names, but got\n{correlations.index.tolist()}\nas the row indices, and\n{X.columns.tolist()}\nas the feature names."
    assert (
        correlations.columns == X.columns
    ).all(), f"The column names should match the feature names, but got\n{correlations.columns.tolist()}\nas the column names, and\n{X.columns.tolist()}\nas the feature names."
    assert (
        np.diag(correlations) == 1
    ).all(), f"The diagonal of the correlation matrix should be 1, as each feature is perfectly correlated with itself, but got\n{np.diag(correlations)}."

    X_pl, _ = synthetic_data_pl_df
    correlations_pl = calculate_all_feature_correlations(X_pl)

    assert isinstance(
        correlations_pl, pd.DataFrame
    ), f"The output should be a pandas DataFrame, but got {type(correlations_pl)}."
    assert (
        correlations_pl.shape == (len(X.columns), len(X.columns))
    ), f"The correlation matrix should be square and match the number of features, but:\nNumber of features: {len(X.columns)}\nShape of the correlation matrix: {correlations_pl.shape}."
    assert (
        correlations_pl.columns == X.columns
    ).all(), f"The column names should match the feature names, but got\n{correlations_pl.columns}\nas the column names, and\n{X.columns}\nas the feature names."
    assert (
        correlations_pl.columns == X.columns
    ).all(), f"The column names should match the feature names, but got\n{correlations_pl.columns}\nas the column names, and\n{X.columns}\nas the feature names."

    X_lazy, _ = synthetic_data_pl_lazy
    correlations_lazy = calculate_all_feature_correlations(X_lazy)

    assert isinstance(
        correlations_lazy, pd.DataFrame
    ), f"The output should be a pandas DataFrame, but got {type(correlations_lazy)}."
    assert (
        correlations_lazy.shape == (len(X.columns), len(X.columns))
    ), f"The correlation matrix should be square and match the number of features, but:\nNumber of features: {len(X.columns)}\nShape of the correlation matrix: {correlations_lazy.shape}."
    assert (
        correlations_lazy.columns == X.columns
    ).all(), f"The column names should match the feature names, but got\n{correlations_lazy.columns}\nas the column names, and\n{X.columns}\nas the feature names."
    assert (
        correlations_lazy.columns == X.columns
    ).all(), f"The column names should match the feature names, but got\n{correlations_lazy.columns}\nas the column names, and\n{X.columns}\nas the feature names."


def test_identify_highly_correlated_pairs(synthetic_data):
    df_pandas, _ = synthetic_data
    correlations = calculate_all_feature_correlations(df_pandas)
    # Apply a threshold that makes sense based on your synthetic data or mock data
    pairs = identify_highly_correlated_pairs(correlations, 0.8)
    # Validate that known highly correlated pairs are identified
    assert ("X1", "X4") in pairs, "Highly correlated pairs should include ('X1', 'X4')"
    assert (
        len(pairs) > 0
    ), "Should identify at least one pair of highly correlated features"


@pytest.fixture
def time_series_data():
    """Create a sample time-series dataset with a fold column."""
    data_size = 100
    folds = np.array([(i % 10) + 1 for i in range(data_size)])
    np.random.default_rng(seed=42).shuffle(folds)
    features = {
        "fold": folds,
        "feature1": np.random.default_rng(seed=42).uniform(0, 1, data_size),
        "feature2": np.random.default_rng(seed=42).uniform(0, 1, data_size),
    }
    df = pd.DataFrame(features)
    y = pd.Series(np.random.default_rng(42).integers(0, 2, size=data_size))
    return df, y


def test_generate_X_y_correct_splits(time_series_data):
    df, y = time_series_data
    generator = generate_X_y(df, y, 5, 9)
    for X_train, _, X_test, _ in generator:
        assert all(
            df.loc[X_train.index, "fold"] < df.loc[X_test.index, "fold"].min()
        ), "Training data folds must be less than test data fold."


def test_generate_X_y_missing_fold_column(time_series_data):
    df, y = time_series_data
    df = df.drop(columns=["fold"])
    with pytest.raises(ValueError, match="DataFrame must contain a 'fold' column."):
        next(generate_X_y(df, y, 5, 9))


def test_generate_X_y_no_test_data(time_series_data):
    df, y = time_series_data
    # Ensure there's no fold 10 data
    df = df[df["fold"] < 10]
    generator = generate_X_y(df, y, 10, 10)
    with pytest.raises(ValueError, match="No test data available for fold 10."):
        next(generator)


def test_generate_X_y_boundary_conditions(time_series_data):
    df, y = time_series_data
    # Check starting and ending bounds
    generator = generate_X_y(df, y, 1, 1)
    X_train, y_train, X_test, y_test = next(generator)
    assert len(X_train) < len(
        df
    ), "Training data should be less than total data for the first fold."
    assert len(X_test) > 0, "Test data should be non-empty for the first fold."


def test_evaluate_feature_removal_impact(time_series_data):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score

    df, y = time_series_data
    model = RandomForestClassifier(n_estimators=10)

    # Assume 'feature2' is not critical and thus removing it shouldn't drastically change the AUC
    score_with, score_without = evaluate_feature_removal_impact(
        df, y, model, "feature2", 5, 9
    )

    assert isinstance(
        score_with, list
    ), f"Expected list for score_with, got {type(score_with)}."
    assert isinstance(
        score_without, list
    ), f"Expected list for score_without, got {type(score_without)}."

    assert all(
        isinstance(score, float) for score in score_with + score_without
    ), "All scores should be floats."


def test_features_neither_should_be_removed():
    # Both features improve the model when removed (negative mean impact difference)
    score_with_1 = [0.75, 0.76, 0.77]
    score_without_1 = [0.74, 0.73, 0.72]
    score_with_2 = [0.75, 0.76, 0.77]
    score_without_2 = [0.74, 0.73, 0.72]

    result = select_feature_to_remove(
        score_with_1,
        score_without_1,
        "feature1",
        score_with_2,
        score_without_2,
        "feature2",
    )
    assert (
        result is None
    ), f"No feature should be removed, both are beneficial:\nFeature 1: {score_with_1} -> {score_without_1}\nFeature 2: {score_with_2} -> {score_without_2}\n\nHowever, got {result} as the feature to remove."


def test_one_feature_should_be_removed():
    # Feature 1 has a negative impact, Feature 2 has a positive impact
    score_with_1 = [0.85, 0.86, 0.87]
    score_without_1 = [0.78, 0.79, 0.80]
    score_with_2 = [0.75, 0.76, 0.77]
    score_without_2 = [0.84, 0.83, 0.82]

    result = select_feature_to_remove(
        score_with_1,
        score_without_1,
        "feature1",
        score_with_2,
        score_without_2,
        "feature2",
    )
    assert (
        result == "feature2"
    ), f"Feature 2 should be removed:\nFeature 1: {score_with_1} (lower bound: {np.mean(score_with_1) - np.std(score_with_1)}) -> {score_without_1} (mean: {np.mean(score_without_1)})\nFeature 2: {score_with_2} (lower bound: {np.mean(score_with_2) - np.std(score_with_2)}) -> {score_without_2} (mean: {np.mean(score_without_2)})\n\nHowever, got {result} as the feature to remove."


def test_both_features_have_negative_impact():
    # Both features have negative impacts but Feature 2 is less detrimental
    score_with_1 = [0.75, 0.76, 0.77]
    score_without_1 = [0.78, 0.79, 0.80]
    score_with_2 = [0.75, 0.76, 0.77]
    score_without_2 = [0.76, 0.77, 0.78]

    mean_with_1 = np.mean(score_with_1)
    std_with_1 = np.std(score_with_1)
    mean_without_1 = np.mean(score_without_1)

    mean_with_2 = np.mean(score_with_2)
    std_with_2 = np.std(score_with_2)
    mean_without_2 = np.mean(score_without_2)

    # Calculate the net improvement threshold, considering one standard deviation
    net_improvement_1 = (mean_without_1 - mean_with_1) - std_with_1
    net_improvement_2 = (mean_without_2 - mean_with_2) - std_with_2

    result = select_feature_to_remove(
        score_with_1,
        score_without_1,
        "feature1",
        score_with_2,
        score_without_2,
        "feature2",
    )
    assert (
        result == "feature1"
    ), f"Both features make the model worse, but Feature 1 should be removed:\nFeature 1:\n=========\nRaw scores with 1: {score_with_1}\nMean score with 1: {mean_with_1}\nSD score with 1: {std_with_1}\nRaw scores without 1: {score_without_1}\nMean score without 1: {mean_without_1}\nNet improvement when removing 1: {net_improvement_1}\n\nFeature 2:\n=========\nRaw scores with 2: {score_with_2}\nMean score with 2: {mean_with_2}\nSD score with 2: {std_with_2}\nRaw scores without 2: {score_without_2}\nMean score without 2: {mean_without_2}\nNet improvement when removing 2: {net_improvement_2}\n\nHowever, got {result} as the feature to remove."


def test_edge_case_close_scores():
    # Scores are very close, testing the updated logic
    score_with_1 = [0.7501, 0.7502, 0.7503]
    score_without_1 = [0.7500, 0.7500, 0.7500]
    score_with_2 = [0.7502, 0.7503, 0.7504]
    score_without_2 = [0.7501, 0.7501, 0.7501]

    result = select_feature_to_remove(
        score_with_1,
        score_without_1,
        "feature1",
        score_with_2,
        score_without_2,
        "feature2",
        tolerance=0.0001,
    )
    assert (
        result is None
    ), f"Feature 1 should be removed, close scores but less improvement:\nFeature 1: {score_with_1} (lower bound: {np.mean(score_with_1) - np.std(score_with_1)}) -> {score_without_1} (mean: {np.mean(score_without_1)})\nFeature 2: {score_with_2} (lower bound: {np.mean(score_with_2) - np.std(score_with_2)}) -> {score_without_2} (mean: {np.mean(score_without_2)})\n\nHowever, got {result} as the feature to remove."


def test_features_with_consistent_improvement():
    score_with_1 = [0.70, 0.72, 0.71]
    score_without_1 = [0.80, 0.82, 0.81]  # Consistent improvement
    score_with_2 = [0.65, 0.67, 0.66]
    score_without_2 = [0.68, 0.69, 0.70]  # Marginal improvement

    result = select_feature_to_remove(
        score_with_1,
        score_without_1,
        "feature1",
        score_with_2,
        score_without_2,
        "feature2",
    )
    assert (
        result == "feature1"
    ), "Feature 1 should be removed due to consistent and significant improvement."


def test_initialization_with_few_features():
    # Test with DataFrame that has fewer than two features
    X = pd.DataFrame({"feature1": np.random.default_rng(42).uniform(0, 1, size=10)})
    y = pd.Series(np.random.default_rng(42).integers(0, 2, size=10))
    model = RandomForestClassifier()
    with pytest.raises(ValueError):
        backward_stepwise_feature_selection(X, y, model)


def test_initialization_with_no_fold_column():
    # Ensure the generate_X_y raises an error if no 'fold' column
    X = pd.DataFrame(
        np.random.default_rng(42).uniform(0, 1, size=(10, 2)),
        columns=["feature1", "feature2"],
    )
    y = pd.Series(np.random.default_rng(42).integers(0, 2, size=10))
    with pytest.raises(ValueError):
        list(generate_X_y(X, y))  # Should raise because 'fold' column is missing


def test_feature_removal__INTEGRATION_TEST():
    # Generate data ensuring every fold contains data
    rng = np.random.default_rng(42)

    # Ensures equal distribution of folds
    fold_numbers = np.tile(np.arange(0, 6), 20)

    # Shuffle to avoid ordered split bias
    np.random.default_rng(42).shuffle(fold_numbers)
    X = pd.DataFrame(
        {
            "fold": fold_numbers,
            "feature1": rng.uniform(
                0, 1, size=120
            ),  # this is the selected high-impact feature
            "feature2": rng.poisson(
                2982349, size=120
            ),  # this is the selected low-impact feature
        }
    )
    X["y"] = (X["feature1"] > 0.5).astype(
        int
    )  # ensure feature1 is important by using it to determine y
    y = X["y"]
    X = X.drop(columns="y")
    model = RandomForestClassifier(random_state=42)

    selected_features = backward_stepwise_feature_selection(
        X, y, model, start_fold=1, end_fold=5
    )
    assert (
        "feature2" not in selected_features
    ), f"Feature 2 should be removed, but got {selected_features} as the selected features."


def test_handling_of_correlated_features():
    rng = np.random.default_rng(42)

    # Generate a dataset with explicitly correlated features
    fold_numbers = np.tile(np.arange(1, 11), 100)

    # Add 100 fold 0 samples to ensure all folds have data
    fold_numbers = np.concatenate([fold_numbers, np.zeros(500)])
    rng.shuffle(fold_numbers)
    base_feature = rng.normal(0, 1, size=1500)

    X = pd.DataFrame(
        {
            "fold": fold_numbers,
            "feature1": base_feature,  # Base feature
            "feature2": base_feature * 1.01
            + rng.normal(0, 0.01, size=1500),  # Almost the same as feature1
            "feature3": base_feature * 0.99
            + rng.normal(0, 0.01, size=1500),  # Almost the same as feature1
            "feature4": rng.lognormal(
                0, 1, size=1500
            ),  # Independent high-impact feature
            "feature5": rng.beta(2, 5, size=1500),  # Another independent feature
        }
    )

    # Target variable not strongly influenced by correlated features to ensure they are deemed less important
    X["y"] = (
        (2 * X["feature4"] + X["feature5"] + rng.normal(0, 1, size=1500)) > 1.5
    ).astype(int)
    y = X["y"]
    X = X.drop(columns="y")
    model = RandomForestClassifier(random_state=42)

    original_features = set(X.columns)

    # Perform feature selection
    selected_features = backward_stepwise_feature_selection(
        X, y, model, start_fold=5, end_fold=9, tolerance=0.01
    )

    removed_features = original_features - set(selected_features)
    retained_features = set(selected_features)

    # Assertions to verify that not all correlated features are retained
    correlated_features = {"feature1", "feature2", "feature3"}
    assert (
        len(removed_features & correlated_features) > 0
    ), f"At least one correlated feature should be removed, but got {removed_features & correlated_features} as the removed features."
    assert (
        len(removed_features & correlated_features) < len(correlated_features)
    ), f"Not all correlated features should be removed, but got {removed_features & correlated_features} as the removed features."
    assert (
        len(retained_features & correlated_features) == 1
    ), f"Only one correlated feature should be retained, but got {retained_features & correlated_features} as the retained features."
