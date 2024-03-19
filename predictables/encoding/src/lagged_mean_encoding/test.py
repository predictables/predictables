"""Create a docstring."""

from predictables.encoding.src.lagged_mean_encoding._dynamic_rolling_mean import (
    DynamicRollingMean,
)
from predictables.encoding.src.lagged_mean_encoding._dynamic_rolling_sum import (
    DynamicRollingSum,
)

import polars as pl

__all__ = ["DynamicRollingMean", "DynamicRollingSum"]

df = pl.scan_parquet("cancer_train.parquet")
