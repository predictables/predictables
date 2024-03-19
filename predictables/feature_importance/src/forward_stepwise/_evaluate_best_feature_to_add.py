import polars as pl
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


def evaluate_best_feature_to_add(
    remaining_features: list,
    selected_features: list,
    sample_X: pl.LazyFrame,
    sample_y: pl.LazyFrame,
) -> str:
    best_feature = None
    best_performance = float(
        "inf"
    )  # Assuming we are using MSE which should be minimized

    for feature in remaining_features:
        features_to_test = [*selected_features, feature]
        model = LinearRegression()

        # Assuming sample_X and sample_y are LazyFrames, we collect them here.
        # In a real-world scenario, consider using a more efficient approach
        X_train = sample_X[features_to_test].collect().to_numpy()
        y_train = sample_y.collect().to_numpy()

        model.fit(X_train, y_train)
        predictions = model.predict(X_train)
        performance = mean_squared_error(y_train, predictions)

        if performance < best_performance:
            best_performance = performance
            best_feature = feature

    return best_feature
