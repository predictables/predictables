"""Fit time series models to the lagged columns of a dataset, and estimate the current value."""

import polars as pl
import polars.selectors as cs
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error
from tqdm import tqdm


class TimeSeriesEncoding:
    """Fit time series models to the lagged columns of a dataset, and estimate the current value.

    This class is designed to work with a dataset that has been preprocessed to include
    lagged columns for a single feature. The goal is to predict the current value of the
    feature based on the lagged values, with the intention of using this current value
    as an encoding for a categorical feature in a machine learning model.
    """

    def __init__(
        self, lf: pl.LazyFrame, model: CatBoostRegressor, max_number_of_lags: int = 12
    ):
        """Initialize the TimeSeriesEncoding object with data, model, and parameters."""
        self.lf = lf
        self.original_lf = lf
        self.model = model
        self.max_number_of_lags = max_number_of_lags
        self.data = self.prep_data()
        self.feature_name = self.get_feature()
        self.lags = self.get_lags()
        self.mse_results = self.fit_models()

    def get_feature(self) -> str:
        """Extract the primary feature name assumed to be the target for prediction."""
        return extract_features(self.lf)[0]

    def get_lags(self) -> list:
        """Determine the number of lags used in the dataset from the column names."""
        return extract_lags(self.lf)

    def prep_data(self) -> np.ndarray:
        """Prepare the data by dropping non-numeric columns and converting to numpy array."""
        return self.lf.drop("index").collect().to_numpy()

    def fit_models(self) -> dict:
        """Fit models with varying numbers of lagged features and perform cross-validation."""
        mse_results = {}
        for lags in range(1, self.max_number_of_lags + 1):
            try:
                data = (
                    self.lf.drop("index")
                    .filter(
                        pl.all_horizontal(
                            [
                                pl.col(
                                    f"logit[MEAN_ENCODED_{self.feature_name}_{lag*30}]"
                                ).is_finite()
                                for lag in range(1, lags + 1)
                            ]
                        )
                    )
                    .collect()
                    .to_numpy()
                )
                _, average_mse = sliding_window_cv(data, self.model, train_size=lags)
                mse_results[lags] = average_mse
            except Exception as e:  # noqa: PERF203
                print(f"Error with lag {lags}: {e}")  # noqa: T201
        return mse_results

    def find_optimal_lags(self) -> int:
        """Determine the optimal number of lags using the MSE results from cross-validation."""
        # Fit the models, get the key with the minimum MSE
        return min(self.mse_results, key=self.mse_results.get)

    def predict_current_value(self) -> np.ndarray:
        """Use the optimal number of lags to fit the model and predict the current value."""
        optimal_lags = self.find_optimal_lags()
        X_train = (
            self.lf.select(
                [
                    f"logit[MEAN_ENCODED_{self.feature_name}_{lag*30}]"
                    for lag in range(optimal_lags + 1, 1, -1)
                ]
            )
            .collect()
            .to_numpy()
        )
        y_train = (
            self.lf.select([f"logit[MEAN_ENCODED_{self.feature_name}_30]"])
            .collect()
            .to_numpy()
        )
        model = self.model.fit(X_train, y_train)

        X_pred = (
            self.lf.select(
                [
                    f"logit[MEAN_ENCODED_{self.feature_name}_{lag*30}]"
                    for lag in range(optimal_lags, 0, -1)
                ]
            )
            .collect()
            .to_numpy()
        )
        return model.predict(X_pred)

    def encode_feature(self) -> pl.DataFrame:
        """Encode the feature using the predicted current values."""
        predictions = self.predict_current_value()
        return pl.concat(
            [
                self.lf.select("index"),
                pl.DataFrame({f"{self.feature_name}_current": predictions}).lazy(),
            ],
            how="horizontal",
        )


class TimeSeriesEncoding:
    """Fit time series models to the lagged columns of a dataset, and estimate the current value.

    This class is designed to work with a dataset that has been preprocessed to include
    lagged columns for a single feature. The goal is to predict the current value of the
    feature based on the lagged values, with the intention of using this current value
    as an encoding for a categorical feature in a machine learning model.
    """

    def __init__(
        self, lf: pl.LazyFrame, model: CatBoostRegressor, max_number_of_lags: int = 12
    ):
        """Initialize the TimeSeriesEncoding object with data, model, and parameters."""
        self.lf = lf
        self.model = model
        self.max_number_of_lags = max_number_of_lags
        self.data = self.prep_data()
        self.feature_name = self.get_feature()
        self.lags = self.get_lags()
        self.mse_results = self.fit_models()

    def get_feature(self) -> str:
        """Extract the primary feature name assumed to be the target for prediction."""
        return extract_features(self.lf)[0]

    def get_lags(self) -> list:
        """Determine the number of lags used in the dataset from the column names."""
        return extract_lags(self.lf)

    def prep_data(self) -> np.ndarray:
        """Prepare the data by dropping non-numeric columns and converting to numpy array."""
        return self.lf.drop("index").collect().to_numpy()

    def fit_models(self) -> dict:
        """Fit models with varying numbers of lagged features and perform cross-validation."""
        mse_results = {}
        for lags in range(1, self.max_number_of_lags + 1):
            _, average_mse = sliding_window_cv(self.data, self.model, train_size=lags)
            mse_results[lags] = average_mse
        return mse_results

    def find_optimal_lags(self) -> int:
        """Determine the optimal number of lags using the MSE results from cross-validation."""
        # Fit the models, get the key with the minimum MSE
        return min(self.mse_results, key=self.mse_results.get)

    def predict_current_value(self) -> np.ndarray:
        """Use the optimal number of lags to fit the model and predict the current value."""
        optimal_lags = self.find_optimal_lags()
        X_train = (
            self.lf.select(
                [
                    f"logit[MEAN_ENCODED_{self.feature_name}_{lag*30}]"
                    for lag in range(optimal_lags + 1, 1, -1)
                ]
            )
            .collect()
            .to_numpy()
        )
        y_train = (
            self.lf.select([f"logit[MEAN_ENCODED_{self.feature_name}_30]"])
            .collect()
            .to_numpy()
        )
        model = self.model.fit(X_train, y_train)

        X_pred = (
            self.lf.select(
                [
                    f"logit[MEAN_ENCODED_{self.feature_name}_{lag*30}]"
                    for lag in range(optimal_lags, 0, -1)
                ]
            )
            .collect()
            .to_numpy()
        )
        return model.predict(X_pred)

    def encode_feature(self) -> pl.DataFrame:
        """Encode the feature using the predicted current values."""
        predictions = self.predict_current_value()
        return pl.concat(
            [
                self.lf.select("index"),
                pl.DataFrame({f"{self.feature_name}_current": predictions}).lazy(),
            ],
            how="horizontal",
        )


def synthetic_lf() -> pl.LazyFrame:
    """Generate a synthetic dataset for testing time series models.

    The first column is generated from a lognormal distribution, and each subsequent column
    is generated as a linear function of the previous column, plus a small random noise term.

    It should be fairly easy to predict the next column in the sequence, so this dataset
    can be used for end-to-end testing of the time series model pipeline.

    Returns
    -------
    pl.LazyFrame
        A Polars LazyFrame with 500 rows and 19 columns.
    """
    df = pl.DataFrame(
        {
            "index": np.arange(500),
            "logit[MEAN_ENCODED_feature1_540]": np.random.default_rng(42).lognormal(
                0, 1, 500
            ),
        }
    ).lazy()

    # loop to create columns with lags 60 - 60*18
    for i in range(30 * 18, 0, -30):
        df = df.with_columns(
            [
                # x
                pl.lit((570 - i) / 30)
                # times 2
                .mul(pl.lit(2))
                # plus beta (eg first column)
                .add(pl.col("logit[MEAN_ENCODED_feature1_540]"))
                # plus epsilon (standard normal error)
                .add(pl.Series(np.random.default_rng(42).normal(0, 1, 500)))
                .alias(f"logit[MEAN_ENCODED_feature1_{i}]")
            ]
        ).cache()

    return df.select(
        ["index", *list(reversed([c for c in df.columns if "logit" in c]))]
    )


def load_and_preprocess_data(filepath: str) -> pl.LazyFrame:
    """Load a parquet file and select only the logit columns.

    Parameters
    ----------
    filepath : str
        The path to the parquet file to load. The file should contain columns with names
        starting with "logit".

    Returns
    -------
    pl.LazyFrame
        A Polars LazyFrame with only the "index" and "logit" columns.
    """
    try:
        lf = pl.scan_parquet(filepath)
        logit_columns = [pl.col(c) for c in lf.select(cs.starts_with("logit")).columns]
        if not logit_columns:
            raise ValueError("No logit columns found in the data.")
        return lf.select([pl.col("index"), *logit_columns])
    except Exception as e:
        print(f"Failed to load or preprocess data: {e!s}")  # noqa: T201
        return None


def extract_features(lf: pl.LazyFrame) -> list:
    """Extract the feature names from a Polars LazyFrame.

    Uses the structure of the column names to extract the feature names.

    Parameters
    ----------
    lf : pl.LazyFrame
        The input Polars LazyFrame.

    Returns
    -------
    list
        A list of feature names.
    """
    # Column names are structured as "logit[MEAN_ENCODING_{FEATURE_NAME}_{LAG}]"
    # Extract the feature name from the column names by splitting on "_"
    return [c.split("_")[2] for c in lf.select(cs.starts_with("logit")).columns]


def extract_lags(lf: pl.LazyFrame) -> list:
    """Extract the lag periods from a Polars LazyFrame.

    Uses the structure of the column names to extract the lag periods.

    Parameters
    ----------
    lf : pl.LazyFrame
        The input Polars LazyFrame.

    Returns
    -------
    list
        A list of lag periods.
    """
    # Column names are structured as "logit[MEAN_ENCODING_{FEATURE_NAME}_{LAG}]"
    # Extract the lag period from the column names by splitting on "_"
    return [
        int(c.split("_")[-1].replace("]", ""))
        for c in lf.select(cs.starts_with("logit")).columns
    ]


def recode_columns(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Use the extracted features and lags to recode the column names."""
    return lf.select(
        [pl.col("index")]
        + [
            pl.col(c).alias(f"{feat}_{int(lag/30)}")
            for c, feat, lag in zip(
                lf.select(cs.starts_with("logit")).columns,
                extract_features(lf),
                extract_lags(lf),
            )
        ]
    )


def prepare_data(filename: str) -> np.ndarray:
    """Prepare the data for time series modeling by converting to a numpy array."""
    lf = load_and_preprocess_data(filename)
    return recode_columns(lf).drop("index").collect().to_numpy()


# Modified cross-validation function to include logging
def sliding_window_cv(
    data: np.ndarray, model: CatBoostRegressor, train_size: int, test_size: int = 1
) -> tuple:
    """Perform sliding window cross-validation on a time series dataset.

    Parameters
    ----------
    data : np.ndarray
        The input time series data, with shape (n_samples, n_features).
    model : BaseEstimator
        A scikit-learn compatible model object.
    train_size : int
        The number of lagged features to use for training.
    test_size : int
        The number of periods to predict, by default 1. This procedure
        is intended for a next-period prediction only.

    Returns
    -------
    tuple
        A tuple containing the list of MSE results for each fold, and the average MSE.
    """
    n_samples, n_features = data.shape
    results = []

    for i in range(n_features - train_size - test_size + 1):
        train_start, train_end = i, i + train_size
        test_index = train_end

        X_train = data[:, train_start:train_end]
        y_train = data[:, test_index - 1]
        X_test = data[:, train_start + 1 : train_end + 1]
        y_test = data[:, test_index]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        results.append(mse)
    print(  # noqa: T201
        f"Averaged MSE for {train_size} lags: {np.mean(results):.4f}"
    )

    return results, np.mean(results)


def fit_models_with_various_lags(data: np.ndarray, model) -> dict:  # noqa: ANN001
    """Fit models using varying numbers of lagged features from 1 to 12 and perform cross-validation.

    Parameters
    ----------
    data : np.ndarray
        The input time series data, with shape (n_samples, n_features).
    model : Any
        A scikit-learn compatible model object to be used for the time series predictions.

    Returns
    -------
    dict
        A dictionary where keys are the number of lags and values are the average MSE for each configuration.
    """
    mse_results = {}
    for lags in range(1, 13):  # Lags from 1 to 12
        # Adjust the model training by setting the train_size to 'lags'
        _, average_mse = sliding_window_cv(data, model, train_size=lags)
        mse_results[lags] = average_mse
        print(f"Completed CV for {lags} lags: Average MSE = {average_mse}")  # noqa: T201

    return mse_results


def fit_rf_models(data: np.ndarray, **kwargs) -> dict:
    """Fit random forest models using varying numbers of lagged features from 1 to 12 and perform cross-validation.

    Parameters
    ----------
    data : np.ndarray
        The input time series data, with shape (n_samples, n_features).
    **kwargs
        Additional keyword arguments to pass to the RandomForestRegressor.

    Returns
    -------
    dict
        A dictionary where keys are the number of lags and values are the average MSE for each configuration.
    """
    return fit_models_with_various_lags(
        data, RandomForestRegressor(random_state=42, **kwargs)
    )


def fit_catboost_models(data: np.ndarray, **kwargs) -> dict:
    """Fit CatBoost models using varying numbers of lagged features from 1 to 12 and perform cross-validation.

    Parameters
    ----------
    data : np.ndarray
        The input time series data, with shape (n_samples, n_features).
    **kwargs
        Additional keyword arguments to pass to the CatBoostRegressor.

    Returns
    -------
    dict
        A dictionary where keys are the number of lags and values are the average MSE for each configuration.
    """
    return fit_models_with_various_lags(
        data, CatBoostRegressor(random_state=42, verbose=False, **kwargs)
    )

def generate_catboost_feature(
    data: np.ndarray, feature_name: str, **kwargs
) -> pl.LazyFrame:
    """Generate predictions for a given feature using CatBoost models.

    First, use cross-validation to determine the appropriate number of lags
    to use, which will be defined as the number whose average MSE is within
    1 SD of the minimum average MSE.

    Then, fit a CatBoost model using the optimal number of lags:
    - Use up to 12 lags depending on the results of the cross-validation.
    - The lag-1 feature is the first in the X matrix, and lags 2-12 are
      subsequent, again depending on the number of lags.
    - The ultimate goal is to predict the lag-0, or current value of the feature.

    Return a polars LazyFrame with 4 columns:
    1. the index column
    2. the actual lag-1 values
    3. the predicted lag-1 values, using the CatBoost model
    4. the predicted lag-0 values, using the CatBoost model

    Parameters
    ----------
    data : np.ndarray
        The input time series data, with shape (n_samples, n_features).
    feature_name : str
        The name of the feature to predict.
    **kwargs
        Additional keyword arguments to pass to the CatBoostRegressor.

    Returns
    -------
    pl.LazyFrame
        A Polars LazyFrame with the index, actual, and predicted values for the feature.
    """
    # Fit Catboost models with varying lags
    mse_results = fit_catboost_models(data, **kwargs)

    # Find the optimal number of lags (smallest number of lags within 1 SD of the minimum MSE)
    min_mse = min(mse_results.values())
    sd = np.std(list(mse_results.values()))
    optimal_lags = min([k for k, v in mse_results.items() if v <= min_mse + sd])

    # Fit the CatBoost model with the optimal number N of lags, using data starting with
    # lag-1 and ending with lag-N as the X matrix, and lag-0 (unknown) as the target
    X_train = data[:-1, :optimal_lags]  # Use the first (N-1) samples for training
    y_train = data[:-1, optimal_lags]  # Predicting the next value

    X_test = data[-1:, :optimal_lags]  # Use the last sample for testing/prediction

    model = CatBoostRegressor(random_state=42, n_jobs=-1, **kwargs)

    model.fit(X_train, y_train)

    # Predict the lag-1 values
    y_pred_lag1 = model.predict(X_train)

    # Predict the lag-0 values
    y_pred_lag0 = model.predict(X_test)

    # Create a Polars LazyFrame with the index, actual, and predicted values
    return pl.DataFrame(
        {
            "index": np.arange(len(data)),
            f"{feature_name}_predicted_lag1": y_pred_lag1,
            f"{feature_name}_predicted_lag0": y_pred_lag0,
        }
    ).lazy()
