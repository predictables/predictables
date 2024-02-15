from datetime import date, datetime

import polars as pl
import pytest

from .._preprocessing import preprocess_data_for_pca


def test_date_column_dropping():
    # Create a mock dataframe with a date column
    mock_data = {
        "date": [datetime(2021, 1, 1), datetime(2021, 1, 2)],
        "value": [1, 2],
    }
    mock_df = pl.DataFrame(mock_data)

    # Process the dataframe
    processed_df = preprocess_data_for_pca(mock_df)

    # Assert that the date column has been dropped
    assert (
        "date" not in processed_df.columns
    ), f"Column 'date' was not dropped -> {processed_df.columns}"


def test_date_column_dropping_v2():
    date_col = pl.date_range(
        date(2021, 1, 1), date(2021, 1, 10), interval="1d", eager=True
    )

    assert (
        date_col.dtype == pl.Date
    ), f"Date column should be pl.Date, got: {date_col.dtype}"

    df = pl.DataFrame(
        {
            "date": date_col,
            "value": range(10),
        }
    )
    assert "date" in df.columns, f"Date column should be present: {df.columns}"
    assert df.select("date").dtypes == [
        pl.Date
    ], f"Date column should be pl.Date: {df.dtypes}"

    processed_df = preprocess_data_for_pca(df)
    assert (
        "date" not in processed_df.columns
    ), f"Date column should be dropped: {df.columns}"


def test_standardizing_numeric_columns():
    df = pl.DataFrame({"numeric_1": [1, 2, 3, 4, 5], "numeric_2": [10, 20, 30, 40, 50]})

    processed_df = preprocess_data_for_pca(df)
    for col in ["numeric_1", "numeric_2"]:
        assert processed_df.select(
            pl.col(col).mean()
        ).collect().item() == pytest.approx(0), f"{col} should have mean close to 0"
        assert processed_df.select(pl.col(col).std()).collect().item() == pytest.approx(
            1
        ), f"{col} should have standard deviation close to 1"


def test_dropping_high_cardinality_categorical_columns():
    df = pl.DataFrame(
        {
            "low_card_cat": ["A", "B", "A", "B", "C"],
            "high_card_cat": ["A", "B", "C", "D", "E"],
        }
    ).select(
        [
            (
                pl.col(c).cast(pl.Categorical).name.keep()
                if pl.__version__ >= "0.19.12"
                else pl.col(c).cast(pl.Categorical).keep_name()
            )
            for c in ["low_card_cat", "high_card_cat"]
        ]
    )

    high_cardinality_threshold = 3
    processed_df = preprocess_data_for_pca(df, high_cardinality_threshold)

    assert (
        "high_card_cat" not in processed_df.columns
    ), f"High cardinality categorical column should be dropped due to {df.select(pl.col('high_card_cat')).n_unique()} unique values being larger than the threshold {high_cardinality_threshold}:\n\n{df.select(pl.col('high_card_cat'))}"
    assert (
        "low_card_cat_A" in processed_df.columns
    ), f"Low cardinality categorical column should not be dropped due to {df.select(pl.col('low_card_cat')).n_unique()} unique values being less than or equal to the threshold {high_cardinality_threshold}:\n\n{df.select(pl.col('low_card_cat'))}"
    assert (
        "low_card_cat_B" in processed_df.columns
    ), f"Low cardinality categorical column should not be dropped due to {df.select(pl.col('low_card_cat')).n_unique()} unique values being less than or equal to the threshold {high_cardinality_threshold}:\n\n{df.select(pl.col('low_card_cat'))}"
    assert (
        "low_card_cat_C" in processed_df.columns
    ), f"Low cardinality categorical column should not be dropped due to {df.select(pl.col('low_card_cat')).n_unique()} unique values being less than or equal to the threshold {high_cardinality_threshold}:\n\n{df.select(pl.col('low_card_cat'))}"
    assert (
        len(processed_df.columns) == 3
    ), f"Processed dataframe should have 3 columns:\n\n{print((f'low_card_cat_{level}' for level in ['A', 'B', 'C']))}"


def test_coding_binary_categorical_columns():
    df = pl.DataFrame(
        {
            "binary_cat": ["yes", "no", "yes", "no", "no"],
            "already_binary": [1, 0, 1, 0, 1],
        }
    ).with_columns(
        [
            pl.col("binary_cat").cast(pl.Categorical),
            pl.col("already_binary").cast(pl.Int8),
        ]
    )

    processed_df = preprocess_data_for_pca(df)

    assert (
        processed_df.select(pl.col("binary_cat"))
        .collect()
        .to_series()
        .unique()
        .to_list()
        .sort()
        == [0, 1].sort()
    ), "Binary categorical column should be coded to 0 and 1"
    assert (
        processed_df.select(pl.col("already_binary"))
        .collect()
        .to_series()
        .unique()
        .to_list()
        .sort()
        == [0, 1].sort()
    ), "Already binary column should remain 0 and 1"


def test_one_hot_encoding_non_binary_categorical_columns():
    df = pl.DataFrame({"cat_col": ["A", "B", "C", "A", "B"]})

    processed_df = preprocess_data_for_pca(df, high_cardinality_threshold=10).collect()

    for value in ["A", "B", "C"]:
        assert (
            f"cat_col_{value}" in processed_df.columns
        ), f"One-hot encoded column 'cat_col_{value}' should be present, but isn't:\n\n{processed_df.columns}"
