"""Logit-transform the columns that were mean-encoded in the prior step."""

from __future__ import annotations
import polars as pl
import os
import shutil
from pathlib import Path
from tqdm import tqdm
from predictables.util.transform import logit_transform


####################################### CONSTANTS
TARGET_COLUMN = "evolve_hit_count"
EXPOSURE_COLUMN = "evolve_quote_count"
DATE_COLUMN = "sub_received_date"
INDEX_COLUMN = "index"
WINDOW = 30
N_COLS = 18

PROJECT_ROOT = "/rdata/aweaver/EGModeling/hit_ratio/bop_model"
########################################


def get_file_list() -> list[str]:
    """Get the list of files in the mean_encoding directory that need to be logit-transformed."""
    # Define the original and backup directories
    path = Path(f"{PROJECT_ROOT}/mean_encoding")
    backup = Path(f"{PROJECT_ROOT}/mean_encoding_backup")

    # Remove the backup directory if it exists
    if backup.exists() and backup.is_dir():
        shutil.rmtree(backup)

    # Create the backup directory
    Path(backup).mkdir(parents=True)

    # Copy the files to the backup directory
    for file in os.listdir(path):
        if file.endswith(".parquet"):
            shutil.copy2(path / file, backup / file)

    # wait for the files to be copied
    while len(os.listdir(backup)) < len(os.listdir(path)):
        pass

    return [
        file
        for file in os.listdir(backup)
        if file.endswith(".parquet")
        and not any(name in file for name in ["train", "val", "test", "backup"])
    ]


def get_data(cat_col: str) -> pl.LazyFrame:
    """Read in the data for the given categorical column from the mean_encoding step."""
    return pl.scan_parquet(
        f"{PROJECT_ROOT}/mean_encoding_backup/{cat_col}_{N_COLS}_lags.parquet"
    )


def lag_to_idx(lag: int) -> int:
    """Convert a lag value to the corresponding time series index."""
    return (N_COLS + 1) - int(lag / WINDOW)


def idx_to_lag(idx: int) -> int:
    """Convert a time series index to the corresponding lag value."""
    return int((N_COLS + 1 - idx) * WINDOW)


def get_column_name(cat_col: str, lag: int) -> str:
    """Get the name of the column that corresponds to the given lag value."""
    return f"ROLLING_MEAN({TARGET_COLUMN}[{cat_col}])[lag:{lag}/win:{WINDOW}]"


def main() -> None:
    """Logit-transform the columns that were mean-encoded in the prior step.

    Steps:
    1. Get the list of files in the mean_encoding directory (exclude any that are 'train', 'val', or 'test').
    2. For each categorical column:
        a. Read in the data from the mean_encoding step.
        b. For each index from 1 to N_COLS:
            i.  logit-transform the column corresponding to that index
            ii. add the transformed column to the LazyFrame
        c. Overwrite the original file with both the original and transformed columns.
    """
    # Get the list of files in the mean_encoding directory and loop through them
    files = get_file_list()
    for file in tqdm(files, total=len(files)):
        # file names follow the convention '{cat_col}_{N_COLS}_lags.parquet'
        cat_col = file.split(f"_{N_COLS}")[0]

        # Wait until the file is copied to the backup directory
        while not Path(f"{PROJECT_ROOT}/mean_encoding_backup/{file}").exists():
            pass
        lf = get_data(cat_col)

        # If the lazyframe already contains logit columns, skip this file
        if any(col.startswith("logit[MEAN_ENCODED") for col in lf.columns):
            continue

        # In each file, loop over the indices and logit-transform the corresponding columns
        for idx in range(1, N_COLS + 1):
            col_name = get_column_name(cat_col, idx_to_lag(idx))
            lf = lf.with_columns([pl.col(col_name).name.keep()]).with_columns(
                [
                    logit_transform(
                        col_name, f"logit[MEAN_ENCODED_{cat_col}_{idx_to_lag(idx)}]"
                    )
                ]
            )

        # Write the transformed data back to the file (same file name)
        lf.collect().write_parquet(
            f"{PROJECT_ROOT}/mean_encoding_backup/{cat_col}_{N_COLS}_lags.parquet"
        )


if __name__ == "__main__":
    main()
