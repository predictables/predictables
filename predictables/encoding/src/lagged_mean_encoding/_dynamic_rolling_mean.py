import polars as pl

from predictables.encoding.src.lagged_mean_encoding._dynamic_rolling_sum import (
    DynamicRollingSum,
)


class DynamicRollingMean(DynamicRollingSum):
    def __init__(self):
        super().__init__()

        self._numrator_col = None
        self._denominator_col = None

    def x_col(*args, **kwargs):
        """
        Warning
        -------
        This method is not implemented for `DynamicRollingMean`. Use
        `numerator_col` and `denominator_col` instead.
        """
        raise NotImplementedError(
            "This method is not implemented for `DynamicRollingMean`. "
            "Use `numerator_col` and `denominator_col` instead."
        )

    def x_name(*args, **kwargs):
        """
        Warning
        -------
        This method is not implemented for `DynamicRollingMean`. Use
        `numerator_col` and `denominator_col` instead.
        """
        raise NotImplementedError(
            "This method is not implemented for `DynamicRollingMean`. "
            "Use `numerator_col` and `denominator_col` instead."
        )

    def numerator_col(self, numerator_col: str) -> "DynamicRollingMean":
        """
        Set the column to use as the numerator in the rolling mean calculation.

        Parameters
        ----------
        numerator_col : str
            The name of the column to use as the numerator in the rolling mean
            calculation.

        Returns
        -------
        DynamicRollingMean
            The `DynamicRollingMean` object with the _numerator_col attribute
            set to the passed value.
        """

        # Define an inside function to use the validate_column decorator
        # @validate_column(self._lf, x_col)
        def _set_numerator_col(self, numerator_col: str) -> "DynamicRollingSum":
            self.numerator_col = numerator_col
            return self

        self._numerator_col = numerator_col
        return self

    def denominator_col(self, denominator_col: str) -> "DynamicRollingMean":
        """
        Set the column to use as the denominator in the rolling mean calculation.

        Parameters
        ----------
        denominator_col : str
            The name of the column to use as the denominator in the rolling mean
            calculation.

        Returns
        -------
        DynamicRollingMean
            The `DynamicRollingMean` object with the _denominator_col attribute
            set to the passed value.
        """

        # Define an inside function to use the validate_column decorator
        # @validate_column(self._lf, x_col)
        def _set_denominator_col(self, denominator_col: str) -> "DynamicRollingSum":
            self.denominator_col = denominator_col
            return self

        self._denominator_col = denominator_col
        return self

    def build_numerator_col(self, lf: pl.LazyFrame) -> None:
        """
        Build the numerator column for the rolling mean calculation, and return
        the `LazyFrame` with the numerator column built.

        Parameters
        ----------
        lf : pl.LazyFrame
            The `LazyFrame` to use to build the numerator column.

        Returns
        -------
        None
            No return value, but the _lf attribute is updated with the
            numerator column built.

        """
        num_lf = (
            DynamicRollingSum()
            .lf(self._lf)
            .date_col(self._date_col)
            .x_col(self._numerator_col)
            .x_name("num")
            .offset(self._offset)
            .window(self._window)
            .category_cols(self._category_cols)
            .index_col(self._index_col)
            .run()
        )

        # Update the _lf attribute
        self._lf = (
            self._lf.join(num_lf, on=self._index_col, how="left")
            .drop("date_right")
            .with_columns([pl.col("rolling_value_list").alias("num")])
            .drop("rolling_value_list")
        )

    def build_denominator_col(self, lf: pl.LazyFrame) -> None:
        """
        Build the denominator column for the rolling mean calculation, and
        update the `LazyFrame` with the denominator column built.

        Parameters
        ----------
        lf : pl.LazyFrame
            The `LazyFrame` to use to build the denominator column.

        Returns
        -------
        None
            No return value, but the _lf attribute is updated with the
            denominator column built.
        """
        den_lf = (
            DynamicRollingSum()
            .lf(self._lf)
            .date_col(self._date_col)
            .x_col(self._denominator_col)
            .x_name("den")
            .offset(self._offset)
            .window(self._window)
            .category_cols(self._category_cols)
            .index_col(self._index_col)
            .run()
        )

        # Update the _lf attribute
        self._lf = (
            self._lf.join(den_lf, on=self._index_col, how="left")
            .drop("date_right")
            .with_columns([pl.col("rolling_value_list").alias("den")])
            .drop("rolling_value_list")
        )

    def run(self) -> pl.LazyFrame:
        """
        Run the rolling mean calculation on the `LazyFrame` and return the
        result.

        Returns
        -------
        pl.LazyFrame
            The `LazyFrame` with the rolling mean calculation applied.
        """
        # Build the numerator and denominator columns, and join them to the
        # original LazyFrame
        self._lf.collect()
        print(self._lf.head().collect())
        self.build_numerator_col(self._lf)
        self._lf.collect()
        print(self._lf.head().collect())
        self.build_denominator_col(self._lf)
        self._lf.collect()
        print(self._lf.head().collect())

        # Calculate the rolling mean
        # mean_col = f"{self._x_name}_mean"
        self._lf = self._lf.with_columns(
            [
                pl.when(pl.col("den") == 0)
                .then(pl.lit(0))
                .otherwise(
                    pl.col("num")
                    .cast(pl.Float64)
                    .truediv(pl.col("den").cast(pl.Float64))
                    .cast(pl.Float64)
                )
                .alias("mean")
            ]
        ).drop(["num", "den"])

        return self._lf
