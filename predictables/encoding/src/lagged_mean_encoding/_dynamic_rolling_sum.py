from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl

from predictables.encoding.src.lagged_mean_encoding._rolling_op_column_name import (
    rolling_op_column_name,
)
from predictables.encoding.src.lagged_mean_encoding.sum.min_date import min_date
from predictables.encoding.src.lagged_mean_encoding.sum.max_date import max_date
from predictables.encoding.src.lagged_mean_encoding.sum.n_dates import n_dates
from predictables.encoding.src.lagged_mean_encoding.sum.format_columns import (
    _format_date_col,
    _format_value_col,
)
from predictables.encoding.src.lagged_mean_encoding.sum.get_date_list_col import (
    _get_date_list_col,
)
from predictables.encoding.src.lagged_mean_encoding.sum.date_list_eval import (
    date_list_eval,
)


class DynamicRollingSum:
    """Create a dynamic rolling sum using polars' lazy API.

    The dynamic rolling sum is a rolling sum that is calculated relative to
    the value in a date column. The rolling sum is calculated for each row
    in the LazyFrame using the values in a value column, and the dates in
    the date column.

    This class uses the builder pattern to set the parameters for the rolling
    sum, and the run method to execute the rolling sum.

    Methods
    -------
    lf(lf: pl.LazyFrame | pl.DataFrame | pd.DataFrame | np.ndarray) -> "DynamicRollingSum":
        Sets the LazyFrame to be used for the rolling sum.
    x_col(x_col: str) -> "DynamicRollingSum":
        Sets the column containing the values that will be summed.
    date_col(date_col: str) -> "DynamicRollingSum":
        Sets the column containing the dates that will be used for
        the rolling sum.
    category_cols(category_cols: str | list[str]) -> "DynamicRollingSum":
        Sets the column containing the categories that will be used to group
        the rolling sum. If a single column is provided, the rolling sum will
        be grouped by that column. If a list of columns is provided, the rolling
        sum will be grouped by the unique combinations of those columns.
        This is also the only optional parameter.
    index_col(index_col: str) -> "DynamicRollingSum":
        Sets the column containing the index that will be used to
        resort the data to the original order after the rolling sum.
    offset(offset: int) -> "DynamicRollingSum":
        Sets the offset for the rolling sum. The offset is the number of days
        before the date in the date column to end the rolling sum. That is,
        it is the number of days to subtract from the date in the date column
        to produce the most recent date to be included in the rolling sum.
    window(window: int) -> "DynamicRollingSum":
        Sets the window for the rolling sum. The window is the number of days
        to include in the rolling sum. The number of days included in the rolling
        sum is exactly the window. The earliest date to be included in the rolling
        sum is the date in the date column minus the sum of the offset and the window.
    rejoin(bool: bool) -> "DynamicRollingSum":
        Sets whether to concatenate the rolling sum to the original LazyFrame.
        If true, returns the original LazyFrame with the rolling sum concatenated
        to the end. If false, returns a LazyFrame with the same number of rows, but
        with columns for the index, date, categories, and rolling sum.
    op(op: str) -> "DynamicRollingSum":
        Sets the operation applied on a rolling basis. This method allows you to
        change the name of the final column. By default, the operation is "ROLLING_SUM",
        and should not be changed in this class, but can be changed in a subclass.
    run() -> pl.LazyFrame:
        Runs the dynamic rolling sum using the provided LazyFrame and parameters.
        First checks that all required parameters have been set, and raises an
        exception if any of the required parameters are not set.
    """

    _lf: pl.LazyFrame | None
    _x_col: str | None
    _x_name: str
    _date_col: str | None
    _category_cols: str | list[str] | None
    _index_col: str | None
    _offset: int | None
    _window: int | None
    _rejoin: bool
    _op: str
    _has_cat_cols: bool

    def __init__(self):
        self._lf = None
        self._x_col = None
        self._x_name = "settoxcol"
        self._date_col = None
        self._category_cols = None
        self._index_col = None
        self._offset = None
        self._window = None
        self._rejoin = False
        self._op = "ROLLING_SUM"
        self._has_cat_cols = False

    def lf(
        self, lf: pl.LazyFrame | pl.DataFrame | pd.DataFrame | np.ndarray
    ) -> "DynamicRollingSum":
        """Set the LazyFrame to be used for the rolling sum.

        The column names should be the same as the column names used
        for the x_col, date_col, and index_col.

        Parameters
        ----------
        lf : pl.LazyFrame | pl.DataFrame | pd.DataFrame | np.ndarray
            The LazyFrame to be used for the rolling sum.

        Returns
        -------
        DynamicRollingSum
            The `DynamicRollingSum` object.
        """
        self._lf = lf
        return self

    def x_col(self, x_col: str = "value") -> "DynamicRollingSum":
        """Set the column to be used for the rolling sum.

        First checks that the column exists in the LazyFrame.

        Parameters
        ----------
        x_col : str
            The name of the column to be used for the rolling sum.

        Returns
        -------
        DynamicRollingSum
            The `DynamicRollingSum` object.
        """

        # Define an inside function to use the validate_column decorator
        # @validate_column(self._lf, x_col)
        def _set_x_col(self, x_col: str) -> "DynamicRollingSum":  # noqa: ANN001
            self._x_col = x_col
            return self  # type: ignore[no-any-return]

        # Return self updated with the validated column
        return _set_x_col(self, x_col)

    def x_name(self, x_name: str | None = None) -> "DynamicRollingSum":
        """Set the name of the column to be used for the rolling sum.

        First checks that the column exists in the LazyFrame.

        Parameters
        ----------
        x_name : str, default None
            The name of the column to be used for the rolling sum. If None, the
            name of the column will be set to the value of the x_col parameter.

        Returns
        -------
        DynamicRollingSum
            The `DynamicRollingSum` object.
        """

        # Define an inside function to use the validate_column decorator
        # @validate_column(self._lf, x_col)
        def _set_x_name(self, x_name: str) -> "DynamicRollingSum":  # noqa: ANN001
            self._x_name = x_name
            return self  # type: ignore[no-any-return]

        # Return self updated with the validated column
        if x_name is not None:
            # change the x_name attribute
            return _set_x_name(self, x_name)
        else:
            # don't make changes
            return self

    def date_col(self, date_col: str = "date") -> "DynamicRollingSum":
        """Set the date column to be used for the rolling sum.

        First checks that the column exists in the LazyFrame.

        Parameters
        ----------
        date_col : str, default "date"
            The name of the date column to be used for the rolling sum.

        Returns
        -------
        DynamicRollingSum
            The `DynamicRollingSum` object.
        """

        # Define an inside function to use the validate_column decorator
        # @validate_column(self._lf, date_col)
        def _set_date_col(self, date_col: str) -> "DynamicRollingSum":  # noqa: ANN001
            self._date_col = date_col
            return self  # type: ignore[no-any-return]

        # Return self updated with the validated column
        return _set_date_col(self, date_col)

    def category_cols(
        self, category_cols: str | list[str] = "category"
    ) -> "DynamicRollingSum":
        """Set the category column to be used for the rolling sum.

        First checks that the column exists in the LazyFrame.

        Parameters
        ----------
        category_cols : str | list[str], default "category"
            The name of the column to be used for the rolling sum.

        Returns
        -------
        DynamicRollingSum
            The `DynamicRollingSum` object.
        """

        # Define an inside function to use the validate_column decorator
        # @validate_column(self._lf, category_cols)
        def _set_category_cols(
            self,  # noqa: ANN001
            category_cols: str | list[str],
        ) -> "DynamicRollingSum":
            self._category_cols = category_cols
            return self  # type: ignore[no-any-return]

        # Return self updated with the validated column
        return _set_category_cols(self, category_cols)

    def index_col(self, index_col: str = "index") -> "DynamicRollingSum":
        """Set the index column to be used for the rolling sum.

        First checks that the column exists in the LazyFrame.

        Parameters
        ----------
        index_col : str, default "index"
            The name of the index column to be used for the rolling sum.

        Returns
        -------
        DynamicRollingSum
            The `DynamicRollingSum` object.
        """

        # Define an inside function to use the validate_column decorator
        # @validate_column(self._lf, index_col)
        def _set_index_col(self, index_col: str) -> "DynamicRollingSum":  # noqa: ANN001
            self._index_col = index_col
            return self  # type: ignore[no-any-return]

        # Return self updated with the validated column
        return _set_index_col(self, index_col)

    def offset(self, offset: int = 30) -> "DynamicRollingSum":
        """Set the offset for the rolling sum.

        The offset is the number of days to offset the rolling sum by.
        The most recent day considered will be the date in the date column minus the offset.

        Parameters
        ----------
        offset : int
            The number of days to offset the rolling sum by.

        Returns
        -------
        DynamicRollingSum
            The `DynamicRollingSum` object.
        """
        self._offset = offset
        return self

    def window(self, window: int = 360) -> "DynamicRollingSum":
        """Set the window for the rolling sum.

        The window is the number of days to include in the rolling sum. The
        earliest day considered will be the date in the date column minus the
        sum of the offset and the window.

        Parameters
        ----------
        window : int
            The number of days to include in the rolling sum.

        Returns
        -------
        DynamicRollingSum
            The `DynamicRollingSum` object.
        """
        self._window = window
        return self

    def rejoin(self, rejoin: bool = False) -> "DynamicRollingSum":
        """Set whether to concatenate the rolling sum to the original LazyFrame.

        If true, returns the original LazyFrame with the rolling sum concatenated
        to the end. If false, returns a LazyFrame with the same number of rows, but
        with columns for the index, date, categories, and rolling sum.

        Parameters
        ----------
        rejoin : bool, default False
            Whether to concatenate the rolling sum to the original LazyFrame.

        Returns
        -------
        DynamicRollingSum
            The `DynamicRollingSum` object.
        """
        self._rejoin = rejoin
        return self

    def op(self, op: str = "ROLLING_SUM") -> "DynamicRollingSum":
        """Set the operation applied on a rolling basis.

        This method allows you to change the name of the final column. By default,
        the operation is "ROLLING_SUM", and should not be changed in this class, but
        can be changed in a subclass.

        Parameters
        ----------
        op : str, optional
            The name of the operation to be applied on a rolling basis. Default
            is "ROLLING_SUM".

        Returns
        -------
        DynamicRollingSum
            The `DynamicRollingSum` object.
        """
        self._op = op
        return self

    def _validate_parameters(self) -> None:
        """Check that all required parameters have been set.

        Raises an exception if any of the required parameters are not set. Does
        not return anything, but raises an exception if any of the required
        parameters are not set.
        """
        required_params = [
            "_lf",
            "_x_col",
            "_date_col",
            "_index_col",
            "_offset",
            "_window",
            "_x_name",
            "_rejoin",
        ]
        for param in required_params:
            if getattr(self, param) is None:
                raise ValueError(f"Parameter {param} has not been set.")

        # If the category columns are set, check that they are in the LazyFrame
        if self._category_cols is not None:
            self._has_cat_cols = True
            if isinstance(self._category_cols, str):
                if self._category_cols not in self._lf.columns:
                    raise ValueError(
                        f"Category column {self._category_cols} not found in LazyFrame. "
                        "Please provide a valid category column."
                    )
                self._category_cols = [self._category_cols]
            else:
                for col in self._category_cols:
                    if col not in self._lf.columns:
                        raise ValueError(
                            f"Category column {col} not found in LazyFrame. "
                            "Please provide a valid category column."
                        )

        # If no category column is set, add a constant column to the LazyFrame to
        # allow the grouped rolling sum to be calculated
        else:
            self._has_cat_cols = False
            self._lf = self._lf.with_columns([pl.lit("0").alias("cat")])
            self._category_cols = ["cat"]

    def _get_parameters(self) -> tuple:
        """Get the parameters for the rolling sum.

        First, validates that all required parameters have been set. Then, returns
        a tuple containing the parameters for the rolling sum class in order:

        - lf
        - x_col
        - x_name
        - date_col
        - category_cols
        - index_col
        - offset
        - window
        - rejoin
        - op

        Returns
        -------
        tuple
            A tuple containing the parameters for the rolling sum class.

        Example
        -------
        >>> d = DynamicRollingSum()
        >>> (lf, x_col, x_name, date, cat, idx, lag, win, rejoin, op) = (
        ...     d._get_parameters()
        ... )
        """
        self._validate_parameters()
        return (
            self._lf if self._lf is not None else pl.LazyFrame(),
            self._x_col if self._x_col is not None else "",
            self._x_name if self._x_name is not None else "",
            self._date_col if self._date_col is not None else "",
            self._category_cols if self._category_cols is not None else "",
            self._index_col if self._index_col is not None else "",
            self._offset if self._offset is not None else 0,
            self._window if self._window is not None else 0,
            self._rejoin,
            self._op,
        )

    def _get_column_name(self, category_column: str | None = None) -> str:
        """Get the name of the column for the rolling sum.

        Returns the name of the column for the rolling sum. If a categorical
        column is provided, the name of the column for the rolling sum will
        be the name of the value column, suffixed with the name of the categorical
        column. If no categorical column is provided, the name of the column for
        the rolling sum will be the name of the value column.

        Parameters
        ----------
        category_column : str, default None
            The name of the categorical column, if any.

        Returns
        -------
        str
            The name of the column for the rolling sum.
        """
        self._validate_parameters()
        (_, x_col, _, _, cat, _, lag, win, _, op) = self._get_parameters()

        # Remove brackets from the category column name if any:
        cat = cat[0] if len(cat) != 1 else cat

        return rolling_op_column_name(op, x_col, cat, lag, win)

    def _get_unique_levels(self) -> list[str] | list[list[str]]:
        """Get the unique levels of the categorical column.

        Returns the unique levels of the categorical column. If no categorical
        column is provided, returns an empty list.

        Returns
        -------
        list[str]
            The unique levels of the categorical column.
        """
        self._validate_parameters()
        (lf, _, _, _, cat, _, _, _, _, _) = self._get_parameters()

        # If there is a categorical column, return the unique levels
        return [
            lf.select([pl.col(col).unique().name.keep()]).collect()[col].to_list()
            for col in cat
        ]

    def _filter_by_level(self, level: str) -> pl.LazyFrame:
        """Filter the LazyFrame by the level of the categorical column.

        Returns a LazyFrame filtered by the level of the categorical column.

        Parameters
        ----------
        level : str
            The level of the categorical column to filter by.

        Returns
        -------
        pl.LazyFrame
            A LazyFrame filtered by the level of the categorical column.
        """
        self._validate_parameters()
        (lf, _, _, _, cat, _, _, _, _, _) = self._get_parameters()

        return lf.filter(pl.col(cat).cast(pl.Utf8) == str(level))

    def _calculate_sum_at_level(self, level: str) -> pl.LazyFrame:
        self._validate_parameters()
        _, x_col, _, date, cat, idx, lag, win, _, _ = self._get_parameters()

        # Filter the lf at the level of the category
        frame = self._filter_by_level(level)

        # Calculate and return the rolling sum at the level of the category
        return (
            dynamic_rolling_sum(frame, x_col, date, idx, lag, win)
            # Rename the sum column
            .with_columns(
                [pl.col("rolling_value_list").alias(self._get_column_name(cat))]
            )
            # Add the level as a column
            .with_columns([pl.lit(str(level)).cast(pl.Categorical).alias(cat)])
        )

    def _calculate_sum(self) -> pl.LazyFrame:
        self._validate_parameters()
        lf, x_col, _, date, cat, idx, lag, win, _, _ = self._get_parameters()
        # Run the dynamic rolling sum at each level of the category
        frames = [
            self._calculate_sum_at_level(level) for level in self._get_unique_levels()
        ]

        # Concatenate the frames
        return pl.concat(frames, how="vertical").select(
            [
                # Index & date columns:
                pl.col(idx).name.keep(),
                pl.col(date).name.keep(),
            ]
            + [
                # Categorical columns:
                pl.col(c).cast(pl.Categorical).name.keep()
                for c in cat
            ]
            + [
                # Rolling sum column:
                pl.col(f"{self._get_column_name(c)}").name.keep()
                for c in cat
            ]
        )

    def _drop_right_suffixes(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        """Drop any columns with "_right" suffixes.

        Returns a LazyFrame with any columns with "_right" suffixes dropped.

        Parameters
        ----------
        lf : pl.LazyFrame
            The input LazyFrame.

        Returns
        -------
        pl.LazyFrame
            A LazyFrame with any columns with "_right" suffixes dropped.
        """
        # Test whether or not any changes need to be made
        has_rights = any(col.endswith("_right") for col in lf.columns)

        # Return the LazyFrame with any columns with "_right" suffixes dropped
        # if there are any, otherwise return the original LazyFrame
        return (
            lf.drop([col for col in lf.columns if col.endswith("_right")])
            if has_rights
            else lf
        )

    def _rejoin_rolling_column(self, lf_with_drs: pl.LazyFrame) -> pl.LazyFrame:
        """Join the rolling sum to the original LazyFrame if rejoin is True.

        Returns the original LazyFrame with the rolling sum concatenated to the end
        if rejoin is True. Returns a LazyFrame with the same number of rows, but
        with columns for the index, date, categories, and rolling sum if rejoin is False.

        Parameters
        ----------
        lf_with_drs : pl.LazyFrame
            The LazyFrame with the rolling sum.

        Returns
        -------
        pl.LazyFrame
            The original LazyFrame with the rolling sum concatenated to the end
            if rejoin is True. A LazyFrame with the same number of rows, but with
            columns for the index, date, categories, and rolling sum if rejoin is False.
        """
        (lf, _, _, _, _, idx, _, _, rejoin, _) = self._get_parameters()
        col_name = self._get_column_name()

        return (
            lf.join(
                lf_with_drs.select(
                    [pl.col(idx).name.keep(), pl.col(col_name).name.keep()]
                ),
                on=idx,
                how="left",
            )
            if rejoin
            else lf_with_drs
        )

    def run(self) -> pl.LazyFrame:
        """Run the dynamic rolling sum using the provided LazyFrame and parameters.

        This method wraps the _run method, which is the method that actually
        does most of the calculation. This method handles the _rejoin attribute
        and returns the appropriate columns.

        Returns
        -------
        pl.LazyFrame
            A LazyFrame with columns that vary depending on the value of self._rejoin.
        """
        # Validate, collect, and calcluate parameters
        self._validate_parameters()

        # Run the dynamic rolling sum if all parameters are set
        lf_with_drs = self._calculate_sum()

        # Drop any columns with "_right" suffixes
        lf_with_drs = self._drop_right_suffixes(lf_with_drs)

        # Join the rolling sum to the original LazyFrame if rejoin is True
        lf = self._rejoin_rolling_column(lf_with_drs)

        # Drop any columns with "_right" suffixes (again, just in case)
        lf = self._drop_right_suffixes(lf)

        # If an additional "cat" column was added, drop it
        if not self._has_cat_cols:
            lf = lf.drop("cat")

        return lf


def dynamic_rolling_sum(
    lf: pl.LazyFrame,
    x_col: str,
    date_col: str,
    index_col: str = "index",
    offset: int = 30,
    window: int = 360,
) -> pl.LazyFrame:
    # Check to ensure there is an index column:
    if index_col not in lf.columns:
        raise ValueError(
            f"Index column {index_col} not found in LazyFrame. "
            "Please provide a valid index column."
        )
    # Data frame with the row index as a column, to resort the data
    # to the original order after the rolling sum
    lf_order = lf.select([pl.col(index_col).alias("index")])

    # Format columns in input LazyFrame
    lf_ = _format_date_col(lf, date_col)
    lf_ = _format_value_col(lf_, x_col)

    # Create a list column with the dates to be used for the
    # rolling sum, optionally grouping by categorical columns
    # if they are provided
    lf_ = _get_date_list_col(lf_, date_col, offset, window)

    dateval = lf_.select([pl.col(date_col), pl.col(x_col)]).unique().sort(date_col)

    lf_ = (
        lf_
        # Calculate the min, max, and count of the date list column
        .with_columns([min_date(), max_date(), n_dates()])
        # Melt the elements of each list in the date_list column to create
        # a new row for each date in each list. Note also this will duplicate
        # the values in the other columns, including the row index and the
        # date column
        .explode("date_list")
        # Create a new column by mapping the list of dates to a list of values
        .with_columns(date_list_eval(pl.col(date_col), pl.col(x_col)))
        .collect()  # Collecting here to avoid an error
        # Take the index, date, and value list columns
        .select([pl.col(index_col), pl.col(date_col), pl.col("value_list")])
        .lazy()
        # Sum up the values by index (representing the original row order)
        .with_columns(
            [pl.col("value_list").sum().over(index_col).name.prefix("rolling_")]
        )
        # Drop the original value list column
        .drop("value_list")
        .unique()
        .sort(index_col)
    )

    # If there is a creeping duplicate, drop it
    if f"{date_col}_right" in lf_.columns:
        lf_ = lf_.drop(f"{date_col}_right")
    if f"{date_col}_left" in lf_.columns:
        lf_ = lf_.drop(f"{date_col}_left")
    if f"{date_col}_right" in lf_order.columns:
        lf_order = lf_order.drop(f"{date_col}_right")
    if f"{date_col}_left" in lf_order.columns:
        lf_order = lf_order.drop(f"{date_col}_left")

    return lf_order.join(lf_, on=index_col, how="left")


