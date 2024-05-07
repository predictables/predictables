"""Collect constant values for the feature selector module."""

MAIN_DATA_FILE = "./temp.parquet"
CV_FOLD_DATA_FILE = "./cv_folds_ts.parquet"
TARGET = "evolve_hit_count"
TRAINING_DATA_FILE = "./X_train_ts_raw.parquet"
CAT_FILES_FOLDER = "./final_encoding"

THREAD_COUNT = -1
RANDOM_SEED = 42
ONE_HOT_MAX_SIZE = 20
IGNORED_FEATURES = ["fold"]

COLS_TO_DROP = [
    "ml_referral_status",
    "ml_referral_status__as_time_series",
    "rated",
    "rated__as_time_series",
    "has_bop",
    "has_bop__as_time_series",
    "ml_referral_status_missing",
    "ml_referral_status_missing__as_time_series",
    "rated_successfully",
    "rated_successfully__as_time_series",
    "ml_stp_ind",
    "ml_stp_ind__as_time_series",
    "eff_period2__as_time_series",
    "eff_period__as_time_series",
]

CATBOOST_HYPERPARAMETERS = {
    "bagging_temperature": 0.21948,
    "depth": 8,
    "iterations": 5000,
    "l2_leaf_reg": 26.66802,
    "leaf_estimation_iterations": 1,
    "learning_rate": 0.04297,
    "min_data_in_leaf": 74,
    "model_size_reg": 795,
    "penalties_coefficient": 641.13784,
    "subsample": 0.64317,
    "rsm": 0.4506828,
    "random_seed": 42,
    "thread_count": -1,
    "one_hot_max_size": 20,
    "ignored_features": ["fold"],
    "bootstrap_type": "MVS",
    "leaf_estimation_method": "Newton",
    "sampling_frequency": "PerTreeLevel",
    "verbose": False,
}
