"""Fit time series models to the lagged columns of a dataset, and estimate the current value."""

import polars as pl
import polars.selectors as cs
import numpy as np
from sklearn import BaseEstimator
from sklearn.ensemble import RandomForestRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error




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
    data: np.ndarray, model: BaseEstimator, train_size: int, test_size: int = 1
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
            f"Fold {i+1}, Train indices: {train_start}-{train_end}, Test index: {test_index}, MSE: {mse}"
        )

    return results, np.mean(results)


def fit_models_with_various_lags(data: np.ndarray, model: BaseEstimator) -> dict:
    """Fit models using varying numbers of lagged features from 1 to 12 and perform cross-validation.

    Parameters
    ----------
    data : np.ndarray
        The input time series data, with shape (n_samples, n_features).
    model : BaseEstimator
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