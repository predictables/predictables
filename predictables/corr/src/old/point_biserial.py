from itertools import product
from typing import Union

import pandas as pd
import polars as pl
import polars.selectors as cs

from predictables.util import Logger, to_pl_lf
from predictables.util.enums import DataType

logger = Logger(__name__).get_logger()


def point_biserial_expression(
    continuous_col: str,
    binary_col: str,
    dtype_continuous: DataType = DataType.CONTINUOUS,
    dtype_binary: DataType = DataType.BINARY,
    df: Union[pl.DataFrame, pl.LazyFrame, pd.DataFrame] = None,
) -> pl.Expr:
    """
    Compute the point-biserial correlation coefficient between a continuous and a binary column.

    Parameters
    ----------
    continuous_col : str
        The name of the continuous column.
    binary_col : str
        The name of the binary column.
    dtype_continuous : DataType, optional
        The data type of the continuous column. It should be CONTINUOUS.
    dtype_binary : DataType, optional
        The data type of the binary column. It should be BINARY.

    Returns
    -------
    pl.Expr
        A polars expression that calculates the point-biserial correlation coefficient.

    References
    ----------
    Should roughly follow the formula here:
        https://en.wikipedia.org/wiki/Point-biserial_correlation_coefficient#Calculation
    """

    if dtype_continuous != DataType.CONTINUOUS or dtype_binary != DataType.BINARY:
        return None
    elif continuous_col == binary_col:
        return None
    elif binary_col > continuous_col:
        return None
    elif (df is not None) and (
        (continuous_col not in df.columns) | (binary_col not in df.columns)
    ):
        return None
    elif (df is not None) and (
        (df.select([continuous_col]).unique().count().collect()[continuous_col] <= 2)
        | (df.select([binary_col]).unique().count().collect()[binary_col] > 2)
    ):
        return None
    else:
        # Calculate the point-biserial correlation using an expression that calls scipy's function
        # Note: This is a placeholder, actual implementation may differ depending on your specific requirements and environment
        return [
            (
                # Mean of the continuous column at each level of the binary column
                pl.col(continuous_col)
                .filter(pl.col(binary_col) == "1")
                .mean()
                .alias(f"M1_{continuous_col}_{binary_col}")
            ),
            (
                # Mean of the continuous column at each level of the binary column
                pl.col(continuous_col)
                .filter(pl.col(binary_col) == "0")
                .mean()
                .alias(f"M0_{continuous_col}_{binary_col}")
            ),
            (
                # Square root of the count of the continuous column at each level of the binary column
                pl.col(continuous_col)
                .filter(pl.col(binary_col) == "1")
                .count()
                .sqrt()
                .alias(f"SQRT[n1_{continuous_col}_{binary_col}]")
            ),
            (
                # Square root of the count of the continuous column at each level of the binary column
                pl.col(continuous_col)
                .filter(pl.col(binary_col) == "0")
                .count()
                .sqrt()
                .alias(f"SQRT[n0_{continuous_col}_{binary_col}]")
            ),
            (
                # Total count of the continuous column
                pl.col(continuous_col)
                .count()
                .alias(f"n_{continuous_col}_{binary_col}")
            ),
            (
                # (Sample) standard deviation of the continuous column
                pl.col(continuous_col)
                .std()
                .alias(f"std_{continuous_col}_{binary_col}")
            ),
        ]


def point_biserial(df: Union[pl.DataFrame, pl.LazyFrame, pd.DataFrame]) -> pl.DataFrame:
    """
    Compute the point-biserial correlation coefficient between all pairs of continuous and binary columns of a DataFrame.

    Parameters
    ----------
    df : pl.DataFrame or pl.LazyFrame or pd.DataFrame
        The DataFrame containing the columns.

    Returns
    -------
    pl.DataFrame
        A DataFrame containing the point-biserial correlation coefficient between all pairs of continuous and binary columns.
    """
    df = to_pl_lf(df)

    corr_query = []
    cols = []
    corr_component_df = df.__deepcopy__()
    for continuous_col, binary_col in product(df.columns, df.columns):
        # Collect the data for both columns
        continuous_data = (
            corr_component_df.select([continuous_col])
            .collect()
            .to_pandas()[continuous_col]
        )
        binary_data = (
            corr_component_df.select([binary_col]).collect().to_pandas()[binary_col]
        )

        if continuous_col not in df.select(cs.numeric()).columns:
            logger.debug(
                f"Skipping column {continuous_col} because it is not numeric:\n\n{df.select([continuous_col]).collect().to_pandas()[continuous_col]}\n\n"
            )
            continue
        else:
            max_diff = continuous_data.max() - continuous_data.min()
            max_unique_diff = (
                continuous_data.drop_duplicates().sort_values().diff().dropna().max()
            )

        # Check if continuous_data is actually numeric
        try:
            continuous_data = continuous_data.astype(float)
        except Exception as e:
            logger.debug(
                f"Error converting continuous column {continuous_col} to float: {e}"
            )
            continue

        # Make sure there are at most 2 unique values in the binary column
        if len(binary_data.unique()) > 2:
            logger.debug(
                f"Skipping using column {binary_col} as BINARY because it has more than 2 unique values:\n\n{binary_data.unique()[:min(len(binary_data.unique()), 5)]}\n\n"
            )
            continue

        # Make sure the maximum difference between values in the continuous column is not exactly 1
        elif max_diff == 1:
            logger.debug(
                f"Skipping using column {continuous_col} as CONTINUOUS because the maximum difference between values is 1:\n\n{df.select([continuous_col]).collect().to_pandas()[continuous_col]}\n\n"
            )
            continue

        elif max_unique_diff == 1:
            logger.debug(
                f"Skipping using column {continuous_col} as CONTINUOUS because the maximum difference between values is 1:\n\n{df.select([continuous_col]).collect().to_pandas()[continuous_col]}\n\n"
            )
            continue

        # Make sure the continuous column is not all the same value
        elif max_diff == 0:
            logger.debug(
                f"Skipping using column {continuous_col} as CONTINUOUS because all values are the same:\n\n{df.select([continuous_col]).collect().to_pandas()[continuous_col]}\n\n"
            )
            continue

        # Otherwise, get the data types of the columns
        dtype_continuous = DataType.from_pandas_series(
            df.select([continuous_col]).collect().to_pandas()[continuous_col]
        )
        dtype_binary = DataType.from_pandas_series(
            df.select([binary_col]).collect().to_pandas()[binary_col]
        )

        logger.debug(
            f"Processing columns: {continuous_col} (dtype: {dtype_continuous}), {binary_col} (dtype: {dtype_binary})"
        )

        if continuous_col == binary_col:
            logger.debug(
                f"Skipping self-correlation for continuous - binary columns: {continuous_col} - {binary_col}"
            )
            continue
        elif dtype_continuous != DataType.CONTINUOUS or dtype_binary != DataType.BINARY:
            logger.debug(
                f"Skipping non-continuous - binary columns: {continuous_col} - {binary_col}"
            )
            continue
        else:
            pass

        if dtype_continuous == DataType.CONTINUOUS and dtype_binary == DataType.BINARY:
            logger.debug(f"continuous_col: {continuous_col}, binary_col: {binary_col}")
            logger.debug(f"continuous_col dtype: {dtype_continuous}")
            logger.debug(f"binary_col dtype: {dtype_binary}")
            ex = point_biserial_expression(
                continuous_col, binary_col, dtype_continuous, dtype_binary
            )

            logger.debug(f"ex: {ex}")
            if ex is not None:
                # Ensure the data types are correct
                try:
                    dtype = DataType.from_pandas_series(
                        df.select([continuous_col])
                        .collect()
                        .to_pandas()[continuous_col]
                    )
                    logger.debug(f"dtype (continuous column): {dtype}")
                except Exception as e:
                    logger.debug(
                        f"Error getting dtype for continuous column: {continuous_col} - {e}"
                    )

                try:
                    dtype = DataType.from_pandas_series(
                        df.select([binary_col]).collect().to_pandas()[binary_col]
                    )
                    logger.debug(f"dtype (binary column): {dtype}")
                except Exception as e:
                    logger.debug(
                        f"Error getting dtype for binary column: {binary_col} - {e}"
                    )
                corr_query.append(ex)
                corr_component_df = corr_component_df.with_columns(ex)
                cols.append(f"{continuous_col}_{binary_col}")
                logger.debug(
                    f"Generated expressions for columns {continuous_col}, {binary_col}: {ex}"
                )

    if not corr_query:
        return pl.DataFrame()

    # Calculate the actual point-biserial correlation coefficient -- this is the part that
    # requires collecting the above pieces (M1, M0, std, etc.) into a single expression
    corr_coef_query = []
    for col in cols:
        logger.debug(f"Calculating point-biserial correlation for: {col}")
        # Extending the list so I can use .select later and only keep the correlation coefficient
        corr_coef_query += [
            (
                (
                    # M1 - M0 / std --
                    #   this is the left-hand fraction
                    (pl.col(f"M1_{col}") - pl.col(f"M0_{col}"))
                    / pl.col(f"std_{col}")
                )
                * (
                    # SQRT[n1 * n0 / (n * (n - 1))] --
                    #   this is the right-hand fraction under the square root
                    (pl.col(f"SQRT[n1_{col}]") * pl.col(f"SQRT[n0_{col}]"))
                    / (pl.col(f"n_{col}") * (pl.col(f"n_{col}") - 1)).sqrt()
                ).alias(f"correlation_{col}")
            ),
        ]

    # Build a data frame with the correlation coefficients
    corr_coef_df = corr_component_df.select(corr_coef_query).collect()
    logger.debug(f"Final correlation DataFrame: {corr_coef_df}")
    return corr_coef_df
