"""Fit time series models to the lagged columns of a dataset, and estimate the current value."""

import pandas as pd
import polars as pl
import polars.selectors as cs
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error

from collections.abc import Generator


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
    lf = pl.scan_parquet(filepath)
    return lf.select(
        [pl.col("index")]
        + [pl.col(c) for c in lf.select(cs.starts_with("logit")).columns]
    )


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


def X_y_generator(
    lf: pl.LazyFrame, feature_name: str, n_folds: int = 5, n_prior_periods: int = 6
) -> Generator:
    """Generate X, y splits for time series cross-validation.

    Parameters
    ----------
    lf : pl.LazyFrame
        The input Polars LazyFrame.
    feature_name : str
        The feature name used in the column names.
    n_folds : int
        The number of folds to use in the time series cross-validation.
    n_prior_periods : int
        The number of prior periods to use as features.

    Yields
    ------
    tuple
        A tuple of X and y splits.
    """

    def col_name(i: int, feature: str) -> str:
        return f"{feature}_{i}"

    def find_first_col_idx(n_folds: int, n_prior_periods: int) -> int:
        return n_prior_periods + n_folds - 1

    for fold in range(n_folds, 0, -1):
        cols = [col_name(fold + i, feature_name) for i in range(n_prior_periods)]
        yield (
            lf.select(cols[1:]).collect().to_numpy(),
            lf.select([cols[0]]).collect().to_numpy(),
        )


# def time_series_cv_corrected(
#     lf: pl.LazyFrame, features: list, max_lag: int, n_splits: int = 5
# ) -> tuple:
#     tscv = TimeSeriesSplit(n_splits=n_splits)
#     results = []
#     for p in range(1, max_lag + 1):
#         mse_scores = []
#         for train_index, test_index in tscv.split(df):
#             train_df, test_df = df.iloc[train_index], df.iloc[test_index]
#             feature_cols = [f'{feature}_{lag}' for feature in features for lag in range(1, max_lag + 1) if f'{feature}_{lag}' in df.columns]
#             X_train = train_df[feature_cols].values
#             X_test = test_df[feature_cols].values
#             if f'{features[0]}_{p}' in df.columns:
#                 y_train = train_df[f'{features[0]}_{p}'].values
#                 y_test = test_df[f'{features[0]}_{p}'].values
#             else:
#                 continue
#             model = RandomForestRegressor(n_estimators=100, random_state=42)
#             model.fit(X_train, y_train)
#             predictions = model.predict(X_test)
#             mse = mean_squared_error(y_test, predictions)
#             mse_scores.append(mse)
#         if mse_scores:
#             avg_mse = np.mean(mse_scores)
#             results.append((p, avg_mse))
#     best_p = min(results, key=lambda x: x[1])[0] if results else None
#     return results, best_p

# # Main execution
# df_synthetic = synthetic_dataframe()
# df_transformed, features, _ = extract_features_and_lags(df_synthetic.copy())
# results, best_p = time_series_cv_corrected(df_transformed, features, max_lag=3, n_splits=3)
# print("Results (Lag, MSE):", results)
# print("Best lag period based on MSE:", best_p)
# df_transformed.head()