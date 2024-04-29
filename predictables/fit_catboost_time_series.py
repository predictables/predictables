"""Logit-transform the columns that were mean-encoded in the prior step."""

from __future__ import annotations
import polars as pl
from catboost import CatBoostRegressor  # type: ignore[untyped-import]
import os
import typing

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

N_CV_FOLDS = 5
########################################


def get_file_list() -> list[str]:
    """Get the list of logit-transformed files in `FOLDER`."""
    return os.listdir(FOLDER)


def get_cat_col_from_filename(file: str) -> str:
    """Get the categorical column name from the filename.

    For example, if the filename is "first_insured_state_id_18_lags.parquet",
    this function will return "first_insured_state_id", because it is splitting
    at the first occurrence of "_18" (as long as the constant `N_COLS` is still
    18). If there is no "_18" in the filename, it will return the filename without
    the extension.
    """
    return (
        file.split(f"_{N_COLS}")[0]
        if file.find(f"_{N_COLS}") != -1
        else file.split(".")[0]
    )


def start_idx_generator(prior_p: int = 6) -> typing.Generator[int]:
    """Generate the starting index for the lagged columns."""
    for i in range(1, N_COLS - prior_p + 1):
        yield i


def idx_to_column_name(idx: int, file: str) -> str:
    """Convert an index to a column name.

    The column name is in the format "logit[MEAN_ENCODED_{cat_col}_{lag}]".
    The filename is parsed to get the categorical column name, and then
    the lag is calculated from the index.

    For example, if the filename is "first_insured_state_id_18_lags.parquet",
    and the index is 6, this function will return
    "logit[MEAN_ENCODED_first_insured_state_id_360]"

    Here
    360 = 12 * 30 (lags are in 30-day periods)

    The lag is calculated as follows:
    12 = 18 - 6
       = N_COLS - idx
    """
    return f"logit[MEAN_ENCODED_{get_cat_col_from_filename(file)}_{idx_to_lag(idx)}]"


def column_name_to_index(column_name: str) -> int:
    """Convert a column name to an index."""
    return int(column_name.split("_")[-1])


def column_name_generator(
    file: str, prior_p: int = 6
) -> typing.Generator[tuple[list[str], str]]:
    """Generate tuples of lagged columns and the target column, for fitting time-series models incrementally."""
    for start_idx in start_idx_generator():
        yield [
            idx_to_column_name(i, file)
            for i in list(range(start_idx, start_idx + prior_p + 1))
        ]


def fit_single_catboost_model_for_single_col(
    file: str, n_prior_periods: int
) -> CatBoostRegressor:
    """Fit a single CatBoost model to the lagged columns, offset by lag."""
    lf = get_data(file)
    lf = lf.with_columns([pl.lit(n_prior_periods)])
    CatBoostRegressor()


def fit_all_catboost_models_for_single_col(file: str) -> dict[str, CatBoostRegressor]:
    """Fit all CatBoost models to the lagged columns for a single categorical column."""
    lf = get_data(file)
    models = {}
    for cols, target_col in column_name_generator(file):
        lf = lf.select([*cols, target_col])
        model = fit_single_catboost_model_for_single_col(file, len(cols) - 1)
        models[target_col] = model
    return models


# def fit_catboost_models_for_all_cols(files: list[str]) -> dict[str, dict[str, CatBoostRegressor]]:


def main() -> None:
    """Fit CatBoost regression models to the lagged logit-transformed columns.

    The most recent mean-encoded categorical column is 30 days before the submission
    recieved date. In order to push the values to the actual date of the submission,
    we do the following:

    1. Fit CatBoost regression models to the lagged logit-transformed columns, using
       1-12 prior 30-day periods of data.
    2. For each fitted model, use time-series cross validation to select the model
       that performs best on the validation set. There are 18 months of data, so we
       can use 5-fold cross validation.
    """
