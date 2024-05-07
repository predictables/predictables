"""Helper functions for the feature selector."""

import pandas as pd  # type: ignore
import polars as pl  # type: ignore
import polars.selectors as cs  # type: ignore
from catboost import CatBoostClassifier  # type: ignore

from feature_selector_constants import (  # type: ignore
    MAIN_DATA_FILE,
    CV_FOLD_DATA_FILE,
    TARGET,
    TRAINING_DATA_FILE,
    COLS_TO_DROP,
)


def X_y_generator(X: pd.DataFrame, y: pd.Series) -> tuple:
    """Generate the training and testing data."""
    for i in range(5, 10):
        train_idx = (
            get_train()
            .with_row_index()
            .filter(pl.col("fold") <= i)
            .select(pl.col("date_order"))
            # .collect()
            .to_numpy()
            .ravel()
        )
        test_idx = (
            get_train()
            .filter(pl.col("fold") == i + 1)
            .select(pl.col("date_order"))
            # .collect()
            .to_numpy()
            .ravel()
        )

        X_train, y_train = X.iloc[train_idx], y[train_idx]
        X_test, y_test = X.iloc[test_idx], y[test_idx]

        yield X_train, y_train, X_test, y_test


def get_df() -> pl.LazyFrame:
    """Get the main data."""
    df = (
        pl.scan_parquet(MAIN_DATA_FILE)
        .drop(cs.temporal())
        .drop(cs.categorical())
        .drop(cs.string())
        .drop(["exp_years_in_file", "sort_order", "date_order"])
    )

    return (
        df.with_columns(
            [
                pl.col(c).cast(pl.Float64).backward_fill()
                for c in df.columns
                if c != TARGET
            ]
        )
        .with_columns(
            [
                ((pl.col(c) - pl.col(c).mean()) / pl.col(c).std())
                for c in df.columns
                if c != TARGET
            ]
        )
        .with_columns(
            [
                pl.col(c).alias(f"{c.replace('_current', '__as_time_series')}")
                for c in df.select(cs.contains("current")).columns
            ]
        )
    )


def get_cv() -> pl.LazyFrame:
    """Get the cross-validation folds."""
    return pl.scan_parquet(CV_FOLD_DATA_FILE).select(
        [pl.col("date_order"), pl.col("fold")]
    )


def get_train() -> pl.LazyFrame:
    """Get the training data."""
    train = pl.scan_parquet(TRAINING_DATA_FILE)

    train = (
        train.with_columns(
            [
                pl.col(c).cast(pl.Date).name.keep()
                for c in train.select(cs.temporal()).columns
            ]
        )
        .with_columns(
            [
                pl.col("sub_received_date").dt.month().alias("month_received"),
                pl.col("sub_received_date").dt.year().alias("year_received"),
                pl.lit(12)
                .mul(pl.col("sub_received_date").dt.year())
                .add(pl.col("sub_received_date").dt.month())
                .alias("month_year_received"),
            ]
        )
        .select(
            [
                pl.col("date_order"),
                pl.col("sub_received_date"),
                pl.col("month_received"),
                pl.col("year_received"),
                pl.col("month_year_received"),
            ]
            + [
                pl.col(c)
                .cast(pl.Utf8)
                .str.replace(".0", "")
                .str.replace("-999", "0")
                .str.replace("0.0", "0")
                .str.replace("1.0", "1")
                .str.replace("-1", "0")
                .str.replace(".0", "0")
                .cast(pl.Categorical)
                .name.keep()
                for c in train.select(cs.categorical()).columns
            ]
        )
        .with_columns(
            [
                pl.col("eff_period")
                .str.replace("before today", "1")
                .str.replace("after today", "0")
                .str.replace("today", "0")
                .cast(pl.Int64)
                .alias("is_before_today"),
                pl.col("cin_incentive_program")
                .str.replace("post-incentive program", "0")
                .str.replace("pre-incentive program", "1")
                .cast(pl.Int64)
                .alias("is_pre_incentive_program"),
                pl.col("unit")
                .str.replace("cccc", "1")
                .str.replace("small business", "0")
                .cast(pl.Int64)
                .alias("is_unit_cccc"),
            ]
        )
        .drop(["eff_period", "cin_incentive_program", "unit"])
        .collect()
        .lazy()
    )

    return train.with_columns(  # type: ignore
        [
            pl.col(col).cast(pl.Int64)
            for col in train.select(
                [
                    pl.col(c)
                    .value_counts(sort=True, parallel=True)
                    .unique()
                    .count()
                    .name.keep()
                    for c in train.select(cs.categorical()).columns
                ]
            )
            .collect()
            .transpose(include_header=True, column_names=["count"])
            .filter(pl.col("count") == 2)
            .select("column")
            .to_series()
            .to_list()
        ]
    ).join(get_cv(), on="date_order", how="left")


def _get_X() -> pd.DataFrame:
    """Help get_X to get the features."""
    return (
        pl.concat([get_df().drop("fold"), get_train()], how="horizontal")
        .drop(["evolve_hit_count", "date_order", "sub_received_date"])
        .collect()
        .to_pandas()
    )


def get_X() -> pd.DataFrame:
    """Get the features."""
    X = pl.from_pandas(_get_X()).lazy()

    for col in [
        col for col in X.select(cs.numeric()).columns if col not in COLS_TO_DROP
    ]:
        if X.select(pl.col(col).skew().abs()).collect().to_series()[0] > 0.5:
            X = X.with_columns(
                [(pl.col(col).add(pl.lit(1))).log().alias(f"log1p[{col}]")]
            ).with_columns(
                [
                    (pl.col(f"log1p[{col}]").sub(pl.col(f"log1p[{col}]").mean()))
                    .truediv(pl.col(f"log1p[{col}]").std())
                    .alias(f"log1p[{col}]")
                ]
            )

        X = X.with_columns(
            [
                (pl.col(col).sub(pl.col(col).mean()))
                .truediv(pl.col(col).std())
                .alias(col)
            ]
        )

    return X.collect().to_pandas()


def get_y() -> pd.Series:
    """Get the target."""
    return (
        get_df().select(["evolve_hit_count"]).collect().to_pandas()["evolve_hit_count"]
    )


def next_gen_gen(
    X: pd.DataFrame, y: pd.Series, stepwise_hyperparameters: dict
) -> tuple:
    """Generate the next generation."""
    xy_gen = X_y_generator(X, y)
    for X_train, y_train, X_test, y_test in xy_gen:
        yield (
            X_train,
            y_train,
            X_test,
            y_test,
            CatBoostClassifier(**stepwise_hyperparameters),
        )
