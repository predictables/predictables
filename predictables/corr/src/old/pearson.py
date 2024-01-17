from itertools import product
from typing import Union

import pandas as pd
import polars as pl

# set up logger
from predictables.util import Logger, to_pl_lf
from predictables.util.enums import DataType

logger = Logger(__name__).get_logger()


def pearson_expression(
    column1: str,
    column2: str,
    dtype1: DataType = DataType.CONTINUOUS,
    dtype2: DataType = DataType.CONTINUOUS,
) -> pl.Expr:
    """
    Compute the Pearson correlation coefficient between two columns of a DataFrame.

    Parameters
    ----------
    df : pl.LazyFrame
        The DataFrame containing the columns.
    column1 : str
        The name of the first column.
    column2 : str
        The name of the second column.
    dtype1 : DataType, optional
        The data type of the first column. If not provided, it will be inferred.
    dtype2 : DataType, optional
        The data type of the second column. If not provided, it will be inferred.

    Returns
    -------
    pl.Expr
        A polars expression that calculates the Pearson correlation coefficient between the two columns.
    """

    if dtype1 != DataType.CONTINUOUS or dtype2 != DataType.CONTINUOUS:
        # Pearson correlation coefficient can only be calculated for
        # continuous columns
        # logger.debug("condition 1")
        return None
    elif column1 > column2:
        # to prevent double computation:
        # logger.debug("condition 2")
        return None
    elif (column1 == column2) and (dtype1 == DataType.CONTINUOUS):
        # Pearson correlation coefficient is always 1 for the same column
        # logger.debug("condition 3")
        return pl.lit(
            pl.Series([1.0], dtype=pl.Float64),
        ).alias(f"correlation_{column1}_{column2}")
    else:
        # logger.debug("condition 4")
        return (
            pl.corr(column1, column2)
            .cast(pl.Float64)
            .alias(f"correlation_{column1}_{column2}")
        )


def pearson(df: Union[pl.DataFrame, pl.LazyFrame, pd.DataFrame]) -> pl.DataFrame:
    """
    Compute the Pearson correlation coefficient between all pairs of columns of a DataFrame.

    Parameters
    ----------
    df : pl.DataFrame or pl.LazyFrame or pd.DataFrame
        The DataFrame containing the columns.

    Returns
    -------
    pl.DataFrame
        A DataFrame containing the Pearson correlation coefficient between all pairs of columns.
    """
    df = to_pl_lf(df)

    corr_query = []
    for col1, col2 in product(df.columns, df.columns):
        logger.debug(f"col1: {col1}, col2: {col2}")
        if col1 != col2:  # Exclude self-correlation checks here
            logger.debug(f"col1 != col2: {col1 != col2}")
            dtype1 = DataType.from_pandas_series(
                df.select([col1]).collect().to_pandas()[col1]
            )
            dtype2 = DataType.from_pandas_series(
                df.select([col2]).collect().to_pandas()[col2]
            )
            logger.debug(f"dtype1: {dtype1}, dtype2: {dtype2}")

            ex = pearson_expression(col1, col2, dtype1, dtype2)
            logger.debug(f"ex: {ex}")
            if ex is not None:
                logger.debug(f"ex is not None: {ex is not None}")
                corr_query.append(ex)
                logger.debug(f"Post-append: {corr_query}")

    if not corr_query:
        logger.debug(f"not corr_query: {not corr_query}")
        return pl.DataFrame()

    corr_df = df.select(corr_query).collect()
    logger.debug(f"corr_df: {corr_df}")
    return corr_df
