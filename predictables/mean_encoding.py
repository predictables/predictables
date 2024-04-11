"""Create a time series of rolling mean-encoded categorical variables.

The constants defined at the beginning are all that are needed to run the script.
"""

from __future__ import annotations
import logging
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import polars as pl
import polars.selectors as cs
from tqdm import tqdm
import typing

sys.path.append("/rdata/aweaver/EGModeling")
from predictables.encoding import DynamicRollingSum

############################## CONSTANTS
TARGET_COLUMN = "evolve_hit_count"
EXPOSURE_COLUMN = "evolve_quote_count"
DATE_COLUMN = "sub_received_date"
INDEX_COLUMN = "index"
WINDOW = 30
N_COLS = 18

PROJECT_ROOT = "/rdata/aweaver/EGModeling/Hit Ratio/bop_model"
########################################


def clean_index_level_0(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Remove the index level 0 column from the LazyFrame if it exists.

    Parameters
    ----------
    lf : pl.LazyFrame
        The LazyFrame to clean.

    Returns
    -------
    pl.LazyFrame
        The cleaned LazyFrame.

    Examples
    --------
    >>> clean_index_level_0(pl.scan_parquet("file.parquet"))
    """
    return lf.drop("__index_level_0__") if "__index_level_0__" in lf.columns else lf


def read(dataset: str) -> pl.LazyFrame:
    """Read the parquet file and return it as a LazyFrame.

    Parameters
    ----------
    dataset : str
        The dataset to read. Must be one of "train", "val", or "test".

    Returns
    -------
    pl.LazyFrame
        The LazyFrame of the dataset.

    Examples
    --------
    >>> read("train")
    """
    lf = pl.scan_parquet(f"{PROJECT_ROOT}/X_{dataset}_ts_raw.parquet")
    date_cols = lf.select(cs.temporal()).columns
    cat_cols = lf.select(cs.categorical()).columns + lf.select(cs.string()).columns
    num_cols = [*lf.select(cs.numeric()).columns, INDEX_COLUMN]

    cv = (
        pl.scan_parquet(f"{PROJECT_ROOT}/cv_folds_ts.parquet")
        if dataset.lower() == "train"
        else read(dataset.lower()).with_columns(
            [pl.lit(dataset.lower()).cast(pl.Utf8).cast(pl.Categorical).alias("fold")]
        )
    )
    concats = (
        [
            clean_index_level_0(
                pl.scan_parquet(f"{PROJECT_ROOT}/X_{dataset}_ts_raw.parquet")
            ),
            clean_index_level_0(cv).select("fold"),
            clean_index_level_0(
                pl.scan_parquet(f"{PROJECT_ROOT}/y_{dataset}_ts_raw.parquet")
            ),
        ]
        if dataset.lower() == "train"
        else [
            clean_index_level_0(
                pl.scan_parquet(f"{PROJECT_ROOT}/X_{dataset}_ts_raw.parquet")
            ),
            pl.scan_parquet(f"{PROJECT_ROOT}/y_{dataset}_ts_raw.parquet"),
        ]
    )

    out = (
        pl.concat(concats, how="horizontal")
        .with_row_index()
        .with_columns([pl.col(c).cast(pl.Date).name.keep() for c in date_cols])
        .with_columns(
            [pl.col(c).cast(pl.Utf8).cast(pl.Categorical).name.keep() for c in cat_cols]
        )
        .with_columns(
            [
                pl.col(c).cast(pl.Float32).replace(-999, np.nan).name.keep()
                for c in num_cols
            ]
        )
        .drop([c for c in num_cols if c.lower().find("log_plusone") > -1])
        .with_columns(
            [
                pl.col(c)
                .cast(pl.Utf8)
                .str.replace("-999", "0")
                .str.replace("-999.0", "0")
                .str.replace("0.0", "0")
                .str.replace("1.0", "1")
                .cast(pl.Categorical)
                .name.keep()
                for c in cat_cols
            ]
        )
        .with_columns(
            [
                pl.col(c)
                .cast(pl.Utf8)
                .str.replace(".0", "")
                .cast(pl.Categorical)
                .name.keep()
                for c in cat_cols
            ]
        )
    )

    return (
        out.select(
            [c for c in out.columns if c.lower().find("__index_level_0__") == -1]
        )
        .with_columns([pl.lit(1).cast(pl.Float64).alias(EXPOSURE_COLUMN)])
        .with_columns(
            [
                pl.col(c).fill_nan(0).cast(pl.UInt64).name.keep()
                for c in num_cols
                if (
                    (
                        c
                        in [
                            INDEX_COLUMN,
                            "acct_numb_of_fulltime_employees",
                            "acct_numb_of_parttime_employees",
                            "acct_total_numb_of_employees",
                            "cin_numb_of_employees",
                            "total_class_cd_policy",
                        ]
                    )
                    or (c.find("_order") > -1)
                    or (c.find("_year") > -1)
                    or (c.find("_count") > -1)
                )
            ]
        )
    )


def rolling_sum(
    lf: pl.LazyFrame, offset: int, cat_col: str | None = None
) -> pl.LazyFrame:
    """Calculate the rolling sum of the target column, optionally grouped by a categorical column.

    Parameters
    ----------
    lf : pl.LazyFrame
        The LazyFrame to calculate the rolling sum on.
    offset : int
        The offset or lag in days.
    cat_col : str, optional
        The categorical column to group by, by default None.

    Returns
    -------
    pl.LazyFrame
        The LazyFrame with the rolling sum column.

    Examples
    --------
    >>> rolling_sum(pl.scan_parquet("file.parquet"), 30, "cat_col")
    """
    out = (
        DynamicRollingSum()
        .lf(lf)
        .x_col(TARGET_COLUMN)
        .date_col(DATE_COLUMN)
        .index_col(INDEX_COLUMN)
        .offset(offset)
        .window(WINDOW)
        .rejoin(True)
    )

    if cat_col is not None:
        out = out.cat_col(cat_col)

    return out.run()


def rolling_count(
    lf: pl.LazyFrame, offset: int, cat_col: str | None = None
) -> pl.LazyFrame:
    """Calculate the rolling count of the target column, optionally grouped by a categorical column.

    Parameters
    ----------
    lf : pl.LazyFrame
        The LazyFrame to calculate the rolling count on.
    offset : int
        The offset or lag in days.
    cat_col : str, optional
        The categorical column to group by, by default None.

    Returns
    -------
    pl.LazyFrame
        The LazyFrame with the rolling count column.

    Examples
    --------
    >>> rolling_count(pl.scan_parquet("file.parquet"), 30, "cat_col")
    """
    out = (
        DynamicRollingSum()
        .lf(lf)
        .x_col(EXPOSURE_COLUMN)
        .date_col(DATE_COLUMN)
        .index_col(INDEX_COLUMN)
        .offset(offset)
        .window(WINDOW)
        .rejoin(True)
        .op("ROLLING_COUNT")
    )

    if cat_col is not None:
        out = out.cat_col(cat_col)

    return out.run()


def rolling_mean(
    lf: pl.LazyFrame, offset: int, cat_col: str | None = None
) -> pl.LazyFrame:
    """Calculate the rolling mean of the target column, optionally grouped by a categorical column.

    Parameters
    ----------
    lf : pl.LazyFrame
        The LazyFrame to calculate the rolling mean on.
    offset : int
        The offset or lag in days.
    cat_col : str, optional
        The categorical column to group by, by default None.

    Returns
    -------
    pl.LazyFrame
        The LazyFrame with the rolling mean column.

    Examples
    --------
    >>> rolling_mean(pl.scan_parquet("file.parquet"), 30, "cat_col")

    Note
    ----
    The function is implemented as follows:
        1. Calculate the rolling sum of the target column.
        2. Calculate the rolling count of the target column.
        3. Join the two tables on the index column.
        4. Apply Laplace smoothing to the rolling sum and count columns.
        5. Calculate the rolling mean column.
    """
    df1 = rolling_sum(lf, offset, cat_col)
    sum_col = df1.columns[-1]

    df2 = rolling_count(lf, offset, cat_col)
    count_col = df2.columns[-1]

    return (
        df1.join(df2.select([INDEX_COLUMN, count_col]), on=INDEX_COLUMN, how="left")
        .with_columns(
            [
                pl.when(pl.col(count_col) == 0)
                .then(pl.lit(0))
                .otherwise(
                    pl.lit(1)
                    .add(pl.col(sum_col))
                    .truediv(pl.lit(1).add(pl.col(count_col)))
                )
                .alias(sum_col.replace("ROLLING_SUM", "ROLLING_MEAN"))
            ]
        )
        .drop([sum_col, count_col])
    )


def gen(
    c: str,
    n: int = N_COLS,
    lf: pl.LazyFrame = read("train"),  # noqa: B008
) -> typing.Generator[pl.LazyFrame]:
    """Return a generator for the LazyFrames of the rolling mean columns at each lag.

    Parameters
    ----------
    c : str
        The categorical column to generate the rolling mean columns for.
    n : int, optional
        The number of lags to generate. Default is given by the constant N_COLS defined
        at the beginning of the script.
    lf : pl.LazyFrame, optional
        The LazyFrame to generate the rolling mean columns on. Default is the training set.

    Returns
    -------
    typing.Generator[pl.LazyFrame]
        The generator of the LazyFrames of the rolling mean columns.

    Examples
    --------
    >>> iterator = gen("cat_col", 18, pl.scan_parquet("file.parquet"))
    >>> next(iterator).head().collect()
    # Returns the first 5 rows of the FIRST LazyFrame in the generator
    >>> next(iterator).head().collect()
    # Returns the first 5 rows of the SECOND LazyFrame in the generator, etc.
    """
    return (
        rolling_mean(lf, lag, c).select(
            [
                INDEX_COLUMN,
                f"ROLLING_MEAN({TARGET_COLUMN}[{c}])[lag:{lag}/win:{WINDOW}]",
            ]
        )
        for lag in [WINDOW * i for i in range(1, n + 1)]
    )


def cat_cols(lf: pl.LazyFrame = read("train")) -> list[str]:  # noqa: B008
    """Return the categorical columns to generate the rolling mean columns for.

    Parameters
    ----------
    lf : pl.LazyFrame, optional
        The LazyFrame to generate the rolling mean columns on. Default is the training set.

    Returns
    -------
    list[str]
        The list of categorical columns.

    Examples
    --------
    >>> df = (
    ...     pl.DataFrame(
    ...         {"a": [1, 2, 3], "b": ["a", "b", "c"], "c": ["x", "y", "z"]}
    ...     )
    ...     .with_columns([pl.col("b").cast(pl.Categorical)])
    ...     .lazy()
    ... )
    >>> cat_cols(df)
    # ["b", "c"]
    """
    cols = lf.select(cs.categorical()).columns + lf.select(cs.string()).columns
    cols_to_drop = ["curr_home_off_rep_full_nm", "cin_agy_contact_phone_numb"]

    return [c for c in cols if c not in cols_to_drop]


def col_name(c: str, lag: int) -> str:
    """Return the name of the rolling mean column.

    Parameters
    ----------
    c : str
        The categorical column to generate the rolling mean column for.
    lag : int
        The lag in days.

    Returns
    -------
    str
        The name of the rolling mean column.

    Examples
    --------
    >>> TARGET_COLUMN = "y_target_test"
    >>> WINDOW = 30
    >>> col_name("cat_col", 30)
    # "ROLLING_MEAN(y_target_test[cat_col])[lag:30/win:30]"

    >>> WINDOW = 60
    >>> col_name("cat_col", 30)
    # "ROLLING_MEAN(y_target_test[cat_col])[lag:30/win:60]"
    """
    return f"ROLLING_MEAN({TARGET_COLUMN}[{c}])[lag:{lag}/win:{WINDOW}]"


def one_category(
    cat_col: str,
    n: int = N_COLS,
    lf: pl.LazyFrame = read("train"),  # noqa: B008
) -> None:
    """Generate time series columns for the rolling-mean-encoded category.

    Parameters
    ----------
    cat_col : str
        The categorical column to generate the rolling mean columns for.
    n : int, optional
        The number of lags to generate. Default is given by the constant N_COLS defined
        at the beginning of the script.
    lf : pl.LazyFrame, optional
        The LazyFrame to generate the rolling mean columns on. Default is the training set.

    Returns
    -------
    None

    Examples
    --------
    >>> one_category("cat_col", 18, pl.scan_parquet("file.parquet"))
    """
    # create a logger object that logs to a file the info and debug messages:
    logging.basicConfig(
        filename=f"{PROJECT_ROOT}/mean_encoding/{cat_col}.log", level=logging.DEBUG
    )
    logging.info(f"Generating rolling mean columns for {cat_col}")
    ts = gen(cat_col, n, lf)

    logging.info(f"Creating temp directory for {cat_col}")
    logging.debug(f"Current directory: {Path.cwd()}")
    logging.debug(f"ls -lh for tempres: {os.popen(f'ls -lh {PROJECT_ROOT}/').read()}")  # noqa: S605
    if not Path(f"{PROJECT_ROOT}/tempres").exists():
        subprocess.call(["mkdir", f"{PROJECT_ROOT}/tempres"])  # noqa: S603, S607

    # Generate all the columns & temp tables
    logging.info(f"Generating temp tables for {cat_col}")
    for i in range(n):
        try:
            lag = WINDOW * (i + 1)
            filename = f"{PROJECT_ROOT}/tempres/temp_{cat_col}_{lag}.parquet"
            new_col = col_name(cat_col, lag)
            logging.info(
                f"Generating temp table for {cat_col} with lag {lag} at {filename} - {new_col}\n"
            )

            next(ts).select([INDEX_COLUMN, new_col]).collect().write_parquet(filename)
        except StopIteration:  # noqa: PERF203
            logging.info(f"StopIteration at {i}\n\n")
            break
        except FileNotFoundError as e:
            logging.error(f"Error at {i}:\n{e}\n\n")
            continue

    # Read the temp tables and join them on the index column
    logging.info(f"Reading temp tables for {cat_col} and joining them")

    logging.info(f"Reading first table for {cat_col}")
    out = pl.scan_parquet(f"{PROJECT_ROOT}/tempres/temp_{cat_col}_{WINDOW}.parquet")
    for i in range(1, N_COLS):
        lag = WINDOW * (i + 1)
        filename = f"{PROJECT_ROOT}/tempres/temp_{cat_col}_{lag}.parquet"
        logging.info(f"Reading table {i+1} for {cat_col} at {filename}")
        logging.debug(f"Current columns:n{out.columns}")
        out = out.join(
            pl.scan_parquet(filename).select([INDEX_COLUMN, col_name(cat_col, lag)]),
            on=INDEX_COLUMN,
            how="left",
        )
        logging.debug(f"Updated columns:\n{out.columns}\n\n")

    # write the final table to the current directory
    logging.info(f"Writing final tablefor {cat_col}")
    subprocess.call(["mkdir", "-p", f"{PROJECT_ROOT}/mean_encoding"])  # noqa: S603, S607
    out.collect().write_parquet(
        f"{PROJECT_ROOT}/mean_encoding/{cat_col}_{n}_lags.parquet"
    )

    # delete the temp directory
    logging.info(f"Deleting temp directory for {cat_col}")
    subprocess.call(["rm", "-rf", f"{PROJECT_ROOT}/tempres"])  # noqa: S603, S607


def all_categories(
    lf: pl.LazyFrame = read("train"),  # noqa: B008
    cat_cols: list[str] = cat_cols(read("train")),  # noqa: B008
) -> None:
    """Generate time series columns for all categorical columns."""
    for c in tqdm(cat_cols):
        print(f"Processing {c}")  # noqa: T201
        try:
            one_category(c, N_COLS, lf)
        except Exception as e:
            print(f"Error at {c}:\n{e}")  # noqa: T201
            continue


def main() -> None:
    """Run the main mean-encoding loop."""
    df_train = read("train")
    df_val = read("val")
    df_test = read("test")

    df_train.collect().write_parquet(f"{PROJECT_ROOT}/mean_encoding/train.parquet")
    df_val.collect().write_parquet(f"{PROJECT_ROOT}/mean_encoding/val.parquet")
    df_test.collect().write_parquet(f"{PROJECT_ROOT}/mean_encoding/test.parquet")

    all_categories(df_train, cat_cols(df_train))
