"""Produce a rolling count of the number of occurrences of a value in a window of rows.

The DynamicRollingCount class subclasses the DynamicRollingSum class and does the following:

1. In the __init__ method, it calls the super().__init__ method with the same arguments.
2. Replaces the "x_col" attribute with a "count_col" that is defined as the literal 1.
3. Summing up all 1's is equivalent to counting the number of occurrences of a value in a window of rows,
    so the "sum_col" attribute is not changed.
4. The rest of the class is unchanged, but the appropriate columns are renamed to reflect the different
    output -- counting vs summing.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import polars as pl
from predictables.encoding.src.lagged_mean_encoding._dynamic_rolling_sum import (
    DynamicRollingSum,
)
from predictables.util import to_pl_lf


class DynamicRollingCount(DynamicRollingSum):
    def __init__(self):
        super().__init__()
        self._x_col = "count"
        self._op = "ROLLING_COUNT"

    def lf(
        self, lf: pl.LazyFrame | pl.DataFrame | pd.DataFrame | np.ndarray
    ) -> "DynamicRollingCount":
        """Set the LazyFrame to be used for the rolling count.

        This method overrides the `lf` method of the `DynamicRollingSum` class,
        automatically adding a column of all 1's to the LazyFrame to be used for
        the rolling count.

        Parameters
        ----------
        lf : pl.LazyFrame | pl.DataFrame | pd.DataFrame | np.ndarray
            The LazyFrame to be used for the rolling count.

        Returns
        -------
        DynamicRollingCount
            The `DynamicRollingCount` object.
        """
        self._lf = to_pl_lf(lf).with_columns([pl.lit(1).alias("count")])
        return self

    def x_col(self, x_col: str | None = None) -> "DynamicRollingCount":  # noqa: ARG002
        """Set the column to be used for the rolling count.

        Overrides the `x_col` method of the `DynamicRollingSum` class to
        automatically set the column to be used for the rolling count to the
        "temp_count_col", which is the column of all 1's added to the LazyFrame
        in the `lf` method.

        Parameters
        ----------
        x_col : str, default None
            The column to be used for the rolling count. This parameter is
            very different from the `x_col` parameter of the `DynamicRollingSum`
            class and is only included for consistency with the `DynamicRollingSum`
            and to ensure the inherited `x_col` method is overridden.

        Returns
        -------
        DynamicRollingCount
            The `DynamicRollingCount` object.
        """
        self._x_col = "count"
        return self

    def x_name(self, x_name: str | None = None) -> "DynamicRollingCount":
        """Set the name of the column to be used for the rolling count.

        Overrides the `x_name` method of the `DynamicRollingSum` class to
        set the final output column name. If the `x_name` parameter has already
        been used for a column name, it will be replaced by the default name, "count".

        Parameters
        ----------
        x_name : str, default None
            The name of the column to be used for the rolling count.

        Returns
        -------
        DynamicRollingCount
            The `DynamicRollingCount` object.
        """
        nm = "count" if x_name in self._lf.columns else x_name
        self._x_name = nm
        return self
