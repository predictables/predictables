"""Logit-transform the columns that were mean-encoded in the prior step."""

from __future__ import annotations
import polars as pl
import os
import shutil
from pathlib import Path
from tqdm import tqdm
from predictables.util.transform import logit_transform
from catboost import CatBoostRegressor
import polars.selectors as cs
from pathlib import Path
import typing
import sys

from predictables.logit_transform import get_data, idx_to_lag

####################################### CONSTANTS
TARGET_COLUMN = "evolve_hit_count"
EXPOSURE_COLUMN = "evolve_quote_count"
DATE_COLUMN = "sub_received_date"
INDEX_COLUMN = "index"
WINDOW = 30
N_COLS = 18

PROJECT_ROOT = "/rdata/aweaver/EGModeling/hit_ratio/bop_model"
FOLDER = f"{PROJECT_ROOT}/mean_encoding_backup"
########################################


def get_file_list() -> list[str]:
    """Get the list of logit-transformed files in `FOLDER`."""
    return [""]


def get_cat_col_from_filename(file: str) -> str:
    """Get the categorical column name from the filename."""
    return file.split(f"_{N_COLS}")[0]


def start_idx_generator(prior_p: int = 6) -> typing.Generator[int]:
    """Generate the starting index for the lagged columns."""
    for i in range(1, N_COLS - prior_p + 1):
        yield i


def idx_to_column_name(idx: int, file: str) -> str:
    """Convert an index to a column name."""
    return f"logit[MEAN_ENCODED_{get_cat_col_from_filename(file)}_{idx_to_lag(idx)}]"


def column_name_generator(
    file: str, prior_p: int = 6
) -> typing.Generator[tuple[list[str], str]]:
    """Generate tuples of lagged columns and the target column, for fitting time-series models incrementally."""
    cat_col = get_cat_col_from_filename(file)
    for start_idx in start_idx_generator():
        yield [
            idx_to_column_name(i, file)
            for i in list(range(start_idx, start_idx + prior_p + 1))
        ]


def main() -> None:
    # parse the command-line args

    
    # Get the list of logit-transformed files
    files = get_file_list()

    for file in tqdm(
        files,
        total=len(files),
        desc="Fitting time-series models to categorical variables.",
    ):
        cat_col = get_cat_col_from_filename(file)
        lf = get_data(file)
