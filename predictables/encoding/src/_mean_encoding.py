from __future__ import annotations
import polars as pl
from predictables.util import tqdm


def mean_encode_df(
    df: pl.DataFrame,
    cat_cols: list[str],
    col1: str,
    col2: str,
    date_col: str,
    date_offset: int = 1,
    drop_cols: bool = True,  # noqa: ARG001
    drop_mean_ratio: bool = True,  # noqa: ARG001
    keep_cat_col: bool = True,  # noqa: ARG001
    laplace_alpha: int = 0,
) -> pl.DataFrame:
    """Mean-encode a list of categorical columns in a dataframe using lazy evaluation and window functions.

    Calculate running sums of `col1` and `col2` up to but not including the date in each row.

    Parameters
    ----------
    df : pl.DataFrame
        The dataframe to mean-encode.
    cat_cols : list[str]
        The list of categorical columns to mean-encode.
    col1 : str
        The name of the first column to use for mean encoding.
    col2 : str
        The name of the second column to use for mean encoding.
    date_col : str
        The name of the date column to use for mean encoding.
    date_offset : int, optional
        The number of days to offset the date column. Default is 1.
    drop_cols : bool, optional
        Whether to drop the intermediate columns. Default is True.
    drop_mean_ratio : bool, optional
        Whether to drop the mean ratio column. Default is True.
    keep_cat_col : bool, optional
        Whether to keep the original categorical column. Default is True.
    laplace_alpha : int, optional
        The Laplace smoothing parameter. Default is 0.

    Returns
    -------
    pl.DataFrame
        The mean-encoded dataframe.
    """
    for c in tqdm(cat_cols, desc="Mean-encoding columns"):
        df = mean_encoding_with_ratio_lazy(
            df,
            c,
            col1=col1,
            col2=col2,
            date_col=date_col,
            date_offset=date_offset,
            drop_cols=True,
            drop_mean_ratio=True,
            keep_cat_col=True,
            laplace_alpha=laplace_alpha,
        )

    return df


def mean_encode_df2(
    df: pl.DataFrame,
    cat_cols: list[str],
    col1: str,
    col2: str,
    date_col: str,
    date_offset: int = 1,
    drop_cols: bool = True,  # noqa: ARG001
    drop_mean_ratio: bool = True,  # noqa: ARG001
    keep_cat_col: bool = True,  # noqa: ARG001
    laplace_alpha: int = 0,
) -> pl.DataFrame:
    for c in tqdm(cat_cols, desc="Mean-encoding columns"):
        df = mean_encoding_with_ratio_lazy(
            df,
            c,
            col1=col1,
            col2=col2,
            date_col=date_col,
            date_offset=date_offset,
            drop_cols=True,
            drop_mean_ratio=True,
            keep_cat_col=True,
            laplace_alpha=laplace_alpha,
        )

    return df


def mean_encoding_with_ratio_lazy(
    df: pl.DataFrame,
    cat_col: str,
    col1: str,
    col2: str,
    date_col: str,
    date_offset: int = 1,
    drop_cols: bool = True,
    drop_mean_ratio: bool = True,
    keep_cat_col: bool = True,
    laplace_alpha: int = 0,
) -> pl.DataFrame:
    """Perform mean encoding on a categorical column using lazy evaluation and window functions.

    Calculate running sums of `col1` and `col2` up to but not including the
    date in each row.

    Parameters
    ----------
    df : pl.DataFrame
        The dataframe to mean-encode.
    cat_col : str
        The name of the categorical column to mean-encode.
    col1 : str
        The name of the numerator column to use for mean encoding.
    col2 : str
        The name of the denominator column to use for mean encoding.
    date_col : str
        The name of the date column to use for mean encoding.
    date_offset : int, optional
        The number of days to offset the date column. Default is 1.
    drop_cols : bool, optional
        Whether to drop the intermediate columns. Default is True.
    drop_mean_ratio : bool, optional
        Whether to drop the mean ratio column. Default is True.
    keep_cat_col : bool, optional
        Whether to keep the original categorical column. Default is True.
    laplace_alpha : int, optional
        The Laplace smoothing parameter. Default is 0.

    Returns
    -------
    pl.DataFrame
        The mean-encoded dataframe.
    """
    # Get the number of unique features in the categorical column
    n_cats = df.select(pl.col(cat_col)).n_unique()

    # Laplace label for new column name - only when laplace_alpha > 0 and
    laplace_label = f"(laplace_alpha={laplace_alpha})" if laplace_alpha > 0 else ""

    # polars>=0.20.5 requires .len() instead of .count()
    # Add a row number column to the dataframe (so you can resort it later)
    if pl.__version__ >= "0.20.5":
        lazy_df = df.lazy().with_columns(pl.arange(0, pl.len()).alias("row_ord"))
    else:
        lazy_df = df.lazy().with_columns(pl.arange(0, pl.count()).alias("row_ord"))

    lazy_df = (
        # Sort by the categorical column and the date column
        lazy_df.sort([cat_col, date_col])
        # Get cumulative sums of col1 and col2 by the categorical column
        .with_columns(
            [
                pl.cum_sum(col1).over(cat_col).alias("sum_col1"),
                pl.cum_sum(col2).over(cat_col).alias("sum_col2"),
            ]
        )
        # Shift the cumulative sums by the date offset and divide to get the mean ratio
        .with_columns(
            [
                (
                    (
                        pl.col("sum_col1").shift(date_offset) + laplace_alpha
                    )  # numerator gets 1 * alpha
                    / (
                        pl.col("sum_col2").shift(date_offset) + (n_cats * laplace_alpha)
                    )  # denominator gets n_cats * alpha
                ).alias("mean_ratio")
            ]
        )
        # Fill in the mean ratio for the first row of each category
        .with_columns(
            [
                pl.when(pl.col("sum_col2").shift(date_offset) == 0)
                .then(
                    pl.col("mean_ratio").shift(-date_offset)
                )  # use current mean_ratio if sum_col2 is 0
                .otherwise(pl.col("mean_ratio"))
                .alias("mean_ratio")
            ]
        )
        # Fill in the mean ratio for the first row of a new category
        # integrate the Laplace smoothing here
        .with_columns(
            [
                # if the category changes - sorted by category so this means it
                # is a new category
                pl.when(pl.col(cat_col).shift(date_offset) != pl.col(cat_col))
                .then(
                    (
                        # numerator gets 1 * alpha
                        pl.col("sum_col1") + laplace_alpha
                    )
                    / (
                        # denominator gets n_cats * alpha
                        pl.col("sum_col2") + (n_cats * laplace_alpha)
                    )
                )
                .otherwise(pl.col("mean_ratio"))
                .alias("mean_ratio")
            ]
        )
        .with_columns(
            [
                pl.when(pl.col("row_ord") == 0)
                .then(
                    (
                        # numerator gets 1 * alpha
                        pl.col("sum_col1") + laplace_alpha
                    )
                    / (
                        # denominator gets n_cats * alpha
                        pl.col("sum_col2") + (n_cats * laplace_alpha)
                    )
                )
                .otherwise(pl.col("mean_ratio"))
                .alias(
                    f"{cat_col}_[mean_encoded]{laplace_label}"
                    if keep_cat_col
                    else cat_col
                )
            ]
        )
        .sort("row_ord")
    )

    if drop_cols:
        lazy_df = lazy_df.drop(["row_ord", "sum_col1", "sum_col2"])

    if drop_mean_ratio:
        lazy_df = lazy_df.drop("mean_ratio")

    return lazy_df.collect()
