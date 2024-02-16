from typing import Union

import pandas as pd
import polars as pl
import polars.selectors as cs

from predictables.util import to_pl_lf


def preprocess_data_for_pca(
    df: Union[pd.DataFrame, pl.DataFrame, pl.LazyFrame],
    high_cardinality_threshold: int = 10,
) -> pl.DataFrame:
    """Preprocesses the dataframe for PCA analysis.

    Parameters
    ----------
    df : Union[pd.DataFrame, pl.DataFrame, pl.LazyFrame]
        Dataframe to preprocess.
    high_cardinality_threshold : int, optional
        Threshold for high cardinality categorical columns, by default 10.

    Returns
    -------
    pl.DataFrame
        Preprocessed dataframe.

    Notes
    -----
    - Drops date columns.
        - Here "date columns" are defined to be columns with polars "temporal" dtype.
        - This is done because PCA is not defined for date columns unless they are
          first converted to numeric columns.
        - I am not defining any particular conversion from date to numeric columns
          here.
            - However iF you preprocess your data in a way that converts date columns
              to numeric columns, you should be able to use this function.
    - Standardizes numeric columns to have mean 0 and standard deviation 1.
        - This is done because PCA is sensitive to the scale of the data:
            - PCA essentially finds the directions of greatest variance in the data.
            - If the data is not standardized, then the directions of greatest
              variance will be dominated by the columns with the largest scale.
                - While strictly speaking this will be the largest variance, it
                  is not necessarily the most interesting variance, and may or
                  may not help us understand the data.
    - Drops high cardinality categorical columns
        - Here "high cardinality" is defined to be cardinality greater than the
          `high_cardinality_threshold`
        - `high_cardinality_threshold` is set to 10 by default.
        - This is because one-hot encoding high cardinality categorical columns
          can lead to a large number of columns, each with a small number of counts.
        - Additionally, we are mean-encoding all categorical features anyway, so
          they will get included in this PCA analysis in the mean-encoded form.
    - Codes binary categorical columns (eg categorical columns with only two
      levels) to 0 and 1.
        - If the levels are already 0 and 1, the column is cast to float.
        - Otherwise, the level with the smaller number of counts is coded to 0 and
          the other to 1.
    - One-hot encodes non-binary categorical columns with cardinality less than or
      equal to `high_cardinality_threshold`.
        - This is because one-hot encoding high cardinality categorical columns can
          lead to a large number of columns, each with a small number of counts.
        - For small numbers of levels, one-hot encoding could be useful for PCA.
        - At the very least, adding small numbers of one-hot encoded columns should
          not hurt the PCA analysis.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     "date": ["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04"],
    ...     "numeric_1": [1, 2, 3, 4],
    ...     "numeric_2": [10, 20, 30, 40],
    ...     "low_card_cat": ["A", "B", "A", "B"],
    ...     "high_card_cat": ["A", "B", "C", "D"]
    ... })
    >>> # Preprocess the dataframe - should drop date column, standardize numeric
    >>> # columns, drop high cardinality categorical column, and one-hot encode low
    >>> # cardinality categorical column
    >>> preprocess_data_for_pca(df, high_cardinality_threshold=3)
            numeric_1  numeric_2  low_card_cat_A  low_card_cat_B
        0   -1.341641  -1.341641             1.0             0.0
        1   -0.447214  -0.447214             0.0             1.0
        2    0.447214   0.447214             1.0             0.0
        3    1.341641   1.341641             0.0             1.0

    >>> # Preprocess the dataframe - should drop date column, standardize numeric
    >>> # columns, and drop both categorical columns, because the high cardinality
    >>> # threshold is 1 and both categorical columns have cardinality greater than 1
    >>> # This is obviously not a useful way to preprocess the data, but it should be
    >>> # an error-free way to use the function were you so inclined
    >>> preprocess_data_for_pca(df, high_cardinality_threshold=1)
            numeric_1  numeric_2
        0   -1.341641  -1.341641
        1   -0.447214  -0.447214
        2    0.447214   0.447214
        3    1.341641   1.341641

    """
    # Convert to polars LazyFrame
    df = to_pl_lf(df)

    # Drop date columns
    df = df.drop(list(df.select(cs.temporal()).columns))

    # Standardize numeric columns
    numeric_cols = df.select(cs.numeric()).columns
    for col in numeric_cols:
        df = df.with_columns(
            ((pl.col(col) - pl.col(col).mean()) / pl.col(col).std()).alias(col)
        )

    # If there are any non-categorical strings, cast them to categorical
    string_cols = df.select(cs.by_dtype(pl.Utf8)).columns
    for col in string_cols:
        if pl.__version__ >= "0.19.12":
            df = df.with_columns(pl.col(col).cast(pl.Categorical).name.keep())
        else:
            df = df.with_columns(pl.col(col).cast(pl.Categorical).keep_name())

    # Code categorical columns depending on cardinality
    categorical_cols = df.select(cs.by_dtype(pl.Categorical)).columns
    for col in categorical_cols:
        df.select(pl.col(col)).collect()
        unique_count = df.select(pl.col(col)).collect().n_unique()
        unique_levels = df.select(pl.col(col)).collect().unique().sort(by=col)
        print(f"Column: {col}, Unique Count: {unique_count}")

        # Drop high cardinality categorical columns
        if unique_count > high_cardinality_threshold:
            df = df.drop(col)
            print(f"Dropping column {col} due to high cardinality: {unique_count}")

        # Code the binary categorical columns to 0 and 1
        elif unique_count == 2:
            # Criterion if the levels are (0 and 1) or ("0" and "1")
            try:
                criterion = set(unique_levels) in ({0, 1}, {1, 0})
            except TypeError:
                try:
                    criterion = set(unique_levels) in ({"0", "1"}, {"1", "0"})
                except TypeError:
                    criterion = False

            # Check if the levels are already 0 and 1, and if so, cast to float
            if criterion:
                if pl.__version__ >= "0.19.12":
                    df = df.with_columns(pl.col(col).cast(pl.Float64).name.keep())
                else:
                    df = df.with_columns(pl.col(col).cast(pl.Float64).keep_name())
                print(f"Binary Column A: {col}, Unique Values: {unique_levels}")

            # Otherwise, make the level with the smaller number of counts 0 and
            # the other 1
            else:
                print(
                    f"Binary Column B: {col}, Unique Values: "
                    f"{df.with_columns(pl.col(col).value_counts())}"
                )

                counts = df.select(pl.col(col)).collect()[col].value_counts()
                print(f"Counts:\n============\n{counts}")

                def new_polars(x, y, counts, col):
                    return counts.select(pl.col(col).cast(pl.Utf8).name.keep()).item(
                        x, y
                    )

                def old_polars(x, y, counts, col):
                    return counts.select(pl.col(col).cast(pl.Utf8).keep_name()).item(
                        x, y
                    )

                df = df.with_columns(
                    [
                        pl.col(col)
                        .cast(pl.Utf8)
                        .str.replace(
                            (
                                f"{new_polars(0, 0, counts, col)}"
                                if pl.__version__ >= "0.19.12"
                                else f"{old_polars(0, 0, counts, col)}"
                            ),
                            "0",
                        )
                        .str.replace(
                            (
                                f"{new_polars(1, 0, counts, col)}"
                                if pl.__version__ >= "0.19.12"
                                else f"{old_polars(1, 0, counts, col)}"
                            ),
                            "1",
                        )
                        .cast(pl.Float64)
                        .alias(col)
                    ]
                )

        # Code the non-binary categorical columns using one-hot encoding
        else:
            print(
                f"One-hot encoding column {col} with "
                f"{unique_count} levels: {unique_levels}"
            )
            return (
                df.with_columns(df.select(pl.col(col)).collect().to_dummies())
                .drop(col)
                .collect()
            )

    return df.collect()
