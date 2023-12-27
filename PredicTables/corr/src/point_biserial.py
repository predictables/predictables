from itertools import product
from typing import Union

import pandas as pd
import polars as pl
import scipy.stats as stats

from PredicTables.util import Logger, to_pl_lf
from PredicTables.util.enums import DataType

logger = Logger(__name__).get_logger()


def point_biserial_expression(
    continuous_col: str,
    binary_col: str,
    dtype_continuous: DataType = DataType.CONTINUOUS,
    dtype_binary: DataType = DataType.BINARY,
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
        return False
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
        dtype_continuous = DataType.from_pandas_series(
            df.select([continuous_col]).collect().to_pandas()[continuous_col]
        )
        dtype_binary = DataType.from_pandas_series(
            df.select([binary_col]).collect().to_pandas()[binary_col]
        )

        if dtype_continuous == DataType.CONTINUOUS and dtype_binary == DataType.BINARY:
            logger.debug(f"continuous_col: {continuous_col}, binary_col: {binary_col}")
            logger.debug(f"continuous_col dtype: {dtype_continuous}")
            logger.debug(f"binary_col dtype: {dtype_binary}")
            ex = point_biserial_expression(
                continuous_col, binary_col, dtype_continuous, dtype_binary
            )

            logger.debug(f"ex: {ex}")
            if ex is not None:
                corr_query.append(ex)
                corr_component_df = corr_component_df.with_columns(ex)
                cols.append(f"{continuous_col}_{binary_col}")

    if not corr_query:
        return pl.DataFrame()

    # Calculate the actual point-biserial correlation coefficient
    corr_coef_query = []
    for col in cols:
        corr_coef_query.append(
            [
                (
                    (
                        # M1 - M0 / std
                        (pl.col(f"M1_{col}") - pl.col(f"M0_{col}"))
                        / pl.col(f"std_{col}")
                    )
                    * (
                        # SQRT[n1 * n0 / (n * (n - 1))]
                        (pl.col(f"SQRT[n1_{col}]") * pl.col(f"SQRT[n0_{col}]"))
                        / (pl.col(f"n_{col}") * (pl.col(f"n_{col}") - 1)).sqrt()
                    ).alias(f"correlation_{col}")
                ),
            ]
        )

    corr_coef_df = corr_component_df.select(corr_coef_query).collect()
