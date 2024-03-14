import datetime
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd  # type: ignore
import polars as pl

from predictables.util import validate_lf


class DynamicRollingSum:
    """
    Class to create a dynamic rolling sum using polars' lazy API. The dynamic
    rolling sum is a rolling sum that is calculated relative to the value in
    a date column. The rolling sum is calculated for each row in the LazyFrame
    using the values in a value column, and the dates in the date column.

    This class uses the builder pattern to set the parameters for the rolling
    sum, and the run method to execute the rolling sum.

    Methods
    -------
    lf(lf: Union[pl.LazyFrame, pl.DataFrame, pd.DataFrame, np.ndarray]) -> "DynamicRollingSum":
        Sets the LazyFrame to be used for the rolling sum.
    x_col(x_col: str) -> "DynamicRollingSum":
        Sets the column containing the values that will be summed.
    date_col(date_col: str) -> "DynamicRollingSum":
        Sets the column containing the dates that will be used for
        the rolling sum.
    category_cols(category_cols: Union[str, List[str]]) -> "DynamicRollingSum":
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
    run() -> pl.LazyFrame:
        Runs the dynamic rolling sum using the provided LazyFrame and parameters.
        First checks that all required parameters have been set, and raises an
        exception if any of the required parameters are not set.
    """

    def __init__(self):
        self._lf = None
        self._x_col = None
        self._x_name = "settoxcol"
        self._date_col = None
        self._category_cols = None
        self._index_col = None
        self._offset = None
        self._window = None

    def lf(
        self, lf: Union[pl.LazyFrame, pl.DataFrame, pd.DataFrame, np.ndarray]
    ) -> "DynamicRollingSum":
        """
        Sets the LazyFrame to be used for the rolling sum. The column names
        should be the same as the column names used for the x_col, date_col,
        and index_col.

        Parameters
        ----------
        lf : Union[pl.LazyFrame, pl.DataFrame, pd.DataFrame, np.ndarray]
            The LazyFrame to be used for the rolling sum.

        Returns
        -------
        DynamicRollingSum
            The `DynamicRollingSum` object.
        """
        self._lf = lf
        return self

    def x_col(self, x_col: str = "value") -> "DynamicRollingSum":
        """
        Sets the column to be used for the rolling sum. First checks that the
        column exists in the LazyFrame.

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
        def _set_x_col(self, x_col: str) -> "DynamicRollingSum":
            self._x_col = x_col
            return self

        # Return self updated with the validated column
        return _set_x_col(self, x_col)

    def x_name(self, x_name: Optional[str] = None) -> "DynamicRollingSum":
        """
        Sets the name of the column to be used for the rolling sum. First checks that
        the column exists in the LazyFrame.

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
        def _set_x_name(self, x_name: str) -> "DynamicRollingSum":
            self._x_name = x_name
            return self

        # Return self updated with the validated column
        if x_name is not None:
            # change the x_name attribute
            return _set_x_name(self, x_name)
        else:
            # don't make changes
            return self

    def date_col(self, date_col: str = "date") -> "DynamicRollingSum":
        """
        Sets the date column to be used for the rolling sum. First checks that
        the column exists in the LazyFrame.

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
        def _set_date_col(self, date_col: str) -> "DynamicRollingSum":
            self._date_col = date_col
            return self

        # Return self updated with the validated column
        return _set_date_col(self, date_col)

    def category_cols(
        self, category_cols: Union[str, List[str]] = "category"
    ) -> "DynamicRollingSum":
        """
        Sets the category column to be used for the rolling sum. First checks
        that the column exists in the LazyFrame.

        Parameters
        ----------
        category_cols : Union[str, List[str]], default "category"
            The name of the column to be used for the rolling sum.

        Returns
        -------
        DynamicRollingSum
            The `DynamicRollingSum` object.
        """

        # Define an inside function to use the validate_column decorator
        # @validate_column(self._lf, category_cols)
        def _set_category_cols(
            self, category_cols: Union[str, List[str]]
        ) -> "DynamicRollingSum":
            self._category_cols = category_cols
            return self

        # Return self updated with the validated column
        return _set_category_cols(self, category_cols)

    def index_col(self, index_col: str = "index") -> "DynamicRollingSum":
        """
        Sets the index column to be used for the rolling sum. First checks that
        the column exists in the LazyFrame.

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
        def _set_index_col(self, index_col: str) -> "DynamicRollingSum":
            self._index_col = index_col
            return self

        # Return self updated with the validated column
        return _set_index_col(self, index_col)

    def offset(self, offset: int = 30) -> "DynamicRollingSum":
        """
        Sets the offset for the rolling sum. The offset is the number of days
        to offset the rolling sum by. The most recent day considered will be
        the date in the date column minus the offset.

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
        """
        Sets the window for the rolling sum. The window is the number of days
        to include in the rolling sum. The earliest day considered will be
        the date in the date column minus the sum of the offset and the window.

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

    def run(self) -> pl.LazyFrame:
        """
        Runs the dynamic rolling sum using the provided LazyFrame and parameters.
        Raises an exception if any of the required parameters are not set.

        If a categorical column is provided, the rolling sum will be grouped by
        the unique levels of that column. If a list of categorical columns is
        provided, the rolling sum will be grouped by the unique combinations of
        those columns.

        Returns
        -------
        pl.LazyFrame
            A LazyFrame with columns for the index and the rolling sum.

        Note that the categorical columns are the only optional parameters.
        """

        def test_param(self, p):
            if getattr(self, p) is None:
                raise ValueError(f"Parameter {p} has not been set.")

        # Check that all required parameters have been set
        for p in ["_lf", "_x_col", "_date_col", "_index_col", "_offset", "_window"]:
            test_param(self, p)

        # Check to see whether a x_name has been set

        if self._category_cols is not None:
            if isinstance(self._category_cols, str):
                self._category_cols = [self._category_cols]

            for c in self._category_cols:
                unique_levels = (
                    self._lf.select(c).unique().collect().to_series().to_list()
                )
                cat_dfs = []
                for level in unique_levels:
                    cat_dfs.append(
                        dynamic_rolling_sum(
                            lf=self._lf.filter(pl.col(c).cast(pl.Utf8) == str(level)),
                            x_col=self._x_col,
                            date_col=self._date_col,
                            index_col=self._index_col,
                            offset=self._offset,
                            window=self._window,
                        )
                        .with_columns([pl.lit(str(level)).alias(c)])
                        .with_columns(
                            pl.col("rolling_value_list").alias(f"rolling_{c}")
                        )
                        .drop(["rolling_value_list"])
                        .sort(["index", c])
                    )

                self._lf = self._lf.join(
                    pl.concat(cat_dfs, how="vertical"), on=["index", c], how="left"
                )

                if "date_right" in self._lf.columns:
                    self._lf = self._lf.drop(["date_right"])

            return self._lf.select(
                [
                    pl.col(self._index_col).name.keep(),
                    pl.col(self._date_col).name.keep(),
                ]
                + [
                    pl.col(c).cast(pl.Utf8).cast(pl.Categorical).name.keep()
                    for c in self._category_cols
                ]
                + [pl.col(f"rolling_{c}").name.keep() for c in self._category_cols]
            )
        else:
            # Run the dynamic rolling sum if all parameters are set
            return dynamic_rolling_sum(
                lf=self._lf,
                x_col=self._x_col,
                date_col=self._date_col,
                index_col=self._index_col,
                offset=self._offset,
                window=self._window,
            )


@validate_lf
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
        lf_.with_columns(
            [
                pl.col("date_list")
                .list.eval(pl.element().min())
                .list.first()
                .alias("min_date"),
                pl.col("date_list")
                .list.eval(pl.element().max())
                .list.first()
                .alias("max_date"),
                pl.col("date_list").list.len().alias("n_dates"),
            ]
        )
        .explode("date_list")
        .with_columns(
            [
                pl.when(
                    (pl.col("date_list") >= pl.col("min_date"))
                    & (pl.col("date_list") <= pl.col("max_date"))
                )
                .then(
                    pl.col("date_list")
                    .dt.to_string("%m/%d/%Y")
                    .replace(
                        old=dateval.collect()
                        .sort(date_col)
                        .select(pl.col(date_col).dt.to_string("%m/%d/%Y").name.keep())
                        .to_series(),
                        new=dateval.collect().sort(date_col).select(x_col).to_series(),
                    )
                )
                .otherwise(pl.lit(0.0))
                .str.to_decimal()
                .fill_null(0.0)
                .alias("value_list")
            ]
        )
        .collect()
        .select([pl.col(index_col), pl.col(date_col), pl.col("value_list")])
        .lazy()
        .with_columns(
            [pl.col("value_list").sum().over(index_col).name.prefix("rolling_")]
        )
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


def _format_date_col(lf: pl.LazyFrame, date_col: str) -> pl.LazyFrame:
    """
    Takes a LazyFrame and the name of the date column, and returns a LazyFrame
    with the date column correctly formatted as a Date.
    """
    return lf.with_columns([pl.col(date_col).cast(pl.Date).name.keep()])


def _format_value_col(lf: pl.LazyFrame, value_col: str) -> pl.LazyFrame:
    """
    Takes a LazyFrame and the name of the value column, and returns a LazyFrame
    with the value column correctly formatted as a Float64.
    """
    return lf.with_columns([pl.col(value_col).cast(pl.Float64).name.keep()])


def _get_date_list_col(
    lf: pl.LazyFrame, date_col: str, offset: int = 30, window: int = 360
) -> pl.LazyFrame:
    """
    Takes a LazyFrame and the name of the date column, and optionally an
    integer representing the offset and window, and returns a LazyFrame with
    a new column containing a list of dates. This list of dates is used to
    calculate the rolling sum:

    1.  The list of dates is used to filter the original LazyFrame to only
        include the rows that are within the window.
    2.  The the incremental sum of the value column is calculated on that
        filtered LazyFrame.

    Parameters
    ----------
    lf : pl.LazyFrame
        The input LazyFrame.
    date_col : str
        The name of the date column.
    offset : int, default 30
        The number of days to offset the rolling sum by.
    window : int, default 360
        The number of days to include in the rolling sum.

    Returns
    -------
    pl.LazyFrame
        A LazyFrame with a new column containing a struct with keys
        for a list of dates.
    """
    # Return a list of dates for each row
    return lf.with_columns(
        [
            pl.date_ranges(
                start=pl.col(date_col)
                - datetime.timedelta(days=offset)
                - datetime.timedelta(days=window)
                + datetime.timedelta(days=1),
                end=pl.col(date_col) - datetime.timedelta(days=offset),
                interval="1d",
            ).alias("date_list")
        ]
    )


def _get_value_map(
    lf: pl.LazyFrame, date_col: str, x_col: str
) -> Dict[datetime.date, float]:
    """
    Produces a dictionary mapping dates to values. This is used to map the
    list of dates to a list of values.

    Parameters
    ----------
    lf : pl.LazyFrame
        The input LazyFrame.
    date_col : str
        The name of the date column.
    x_col : str
        The name of the value column.

    Returns
    -------
    Dict[datetime.date, float]
        A dictionary mapping dates to values.
    """
    return {
        d: v for d, v in lf.select([pl.col(date_col), pl.col(x_col)]).collect().rows()
    }


def _handle_date_list(
    lf: pl.LazyFrame, x_col: str, date_col: str, index_col: str
) -> pl.LazyFrame:
    return (
        lf
        # Melt the elements of each list in the date_list column to create
        # a new row for each date in each list. Note also this will duplicate
        # the values in the other columns, including the row index and the
        # date column
        .explode("date_list")
        # Create a new column with the value for each date in the list
        .with_columns(_date_list_eval(_get_value_map(lf, date_col, x_col)))
        # Drop the date list column -- at this point we have substituted
        # each date in the list with the sum value for that date
        .drop("date_list")
        # Sum up the values by index (representing the original row order)
        # and date
        .group_by(index_col)
        .agg(pl.sum("value_list").alias(f"rolling_{x_col}"))
        # Resort the data to the original order
        .sort(index_col)
    )


def _date_list_eval(value_map: dict) -> pl.Expr:
    """
    Takes a dictionary mapping dates to values and returns a function that
    can be used to map a list of dates to a list of values.
    """
    return (
        pl.when(pl.col("date_list").is_in(list(value_map.keys())))
        .then(
            pl.col("date_list").replace(
                old=pl.Series(value_map.keys()), new=pl.Series(value_map.values())
            )
        )
        .otherwise(pl.lit(0))
        .alias("value_list")
    )
