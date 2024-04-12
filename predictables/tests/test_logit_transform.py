import pytest
from predictables.logit_transform import (
    # Functions:
    lag_to_idx,
    idx_to_lag,
    get_column_name,
    # Constants:
    WINDOW,
    N_COLS,
    TARGET_COLUMN,
)


@pytest.mark.parametrize(
    "lag,expected",
    [
        (30, 18),
        (60, 17),
        (90, 16),
        (120, 15),
        (150, 14),
        (180, 13),
        (210, 12),
        (240, 11),
        (270, 10),
        (300, 9),
        (330, 8),
        (360, 7),
        (390, 6),
        (420, 5),
        (450, 4),
        (480, 3),
        (510, 2),
        (540, 1),
    ],
)
def test_lag_to_idx(lag, expected):
    """Test that the lag_to_idx function correctly converts a lag value to the corresponding time series index.

    The expected values were calculated with this code:
    ```python
    lags = [30 * i for i in range(1, 19)]
    ids = list(reversed(list(range(1, 19))))
    [(l, i) for l, i in zip(lags, ids)]
    ```
    """
    assert (
        lag_to_idx(lag) == expected
    ), f"Expected lag {lag} to be converted to index {expected} with window size {WINDOW} and number of columns {N_COLS}."


@pytest.mark.parametrize(
    "idx,expected",
    [
        (18, 30),
        (17, 60),
        (16, 90),
        (15, 120),
        (14, 150),
        (13, 180),
        (12, 210),
        (11, 240),
        (10, 270),
        (9, 300),
        (8, 330),
        (7, 360),
        (6, 390),
        (5, 420),
        (4, 450),
        (3, 480),
        (2, 510),
        (1, 540),
    ],
)
def test_idx_to_lag(idx, expected):
    """Test that the idx_to_lag function correctly converts a time series index to the corresponding lag value.

    The expected values were calculated with this code:
    ```python
    lags = [30 * i for i in range(1, 19)]
    ids = list(reversed(list(range(1, 19))))
    [(i, l) for l, i in zip(lags, ids)]
        ```
    """
    assert (
        idx_to_lag(idx) == expected
    ), f"Expected index {idx} to be converted to lag {expected} with window size {WINDOW} and number of columns {N_COLS}."


@pytest.mark.parametrize(
    "cat_col,lag,expected",
    [
        ("unit", 30, "ROLLING_MEAN(evolve_hit_count[unit])[lag:30/win:30]"),
        ("unit", 60, "ROLLING_MEAN(evolve_hit_count[unit])[lag:60/win:30]"),
        ("unit", 90, "ROLLING_MEAN(evolve_hit_count[unit])[lag:90/win:30]"),
        ("unit", 120, "ROLLING_MEAN(evolve_hit_count[unit])[lag:120/win:30]"),
        ("unit", 150, "ROLLING_MEAN(evolve_hit_count[unit])[lag:150/win:30]"),
        ("unit", 180, "ROLLING_MEAN(evolve_hit_count[unit])[lag:180/win:30]"),
        ("unit", 210, "ROLLING_MEAN(evolve_hit_count[unit])[lag:210/win:30]"),
        ("unit", 240, "ROLLING_MEAN(evolve_hit_count[unit])[lag:240/win:30]"),
        ("unit", 270, "ROLLING_MEAN(evolve_hit_count[unit])[lag:270/win:30]"),
        ("unit", 300, "ROLLING_MEAN(evolve_hit_count[unit])[lag:300/win:30]"),
        ("unit", 330, "ROLLING_MEAN(evolve_hit_count[unit])[lag:330/win:30]"),
        ("unit", 360, "ROLLING_MEAN(evolve_hit_count[unit])[lag:360/win:30]"),
        ("unit", 390, "ROLLING_MEAN(evolve_hit_count[unit])[lag:390/win:30]"),
        ("unit", 420, "ROLLING_MEAN(evolve_hit_count[unit])[lag:420/win:30]"),
        ("unit", 450, "ROLLING_MEAN(evolve_hit_count[unit])[lag:450/win:30]"),
        ("unit", 480, "ROLLING_MEAN(evolve_hit_count[unit])[lag:480/win:30]"),
        ("unit", 510, "ROLLING_MEAN(evolve_hit_count[unit])[lag:510/win:30]"),
        ("unit", 540, "ROLLING_MEAN(evolve_hit_count[unit])[lag:540/win:30]"),
    ],
)
def test_get_column_name(cat_col, lag, expected):
    """Test that the get_column_name function correctly constructs the column name for a given categorical column and lag value.

    The expected values were calculated with this script:
    ```python
    import polars as pl

    lf = pl.scan_parquet(
        "/rdata/aweaver/EGModeling/hit_ratio/bop_model/mean_encoding/unit_18_lags.parquet"
    )
    expecteds = lf.columns[1:]
    lags = [int(e.split("lag:")[1].split("/")[0]) for e in expecteds]
    cols = ["unit" for _ in expecteds]

    [(c, l, e) for c, l, e in zip(cols, lags, expecteds)]
    ```
    """
    assert (
        get_column_name(cat_col, lag) == expected
    ), f"Expected column name {expected} for categorical column {cat_col} and lag {lag}."
