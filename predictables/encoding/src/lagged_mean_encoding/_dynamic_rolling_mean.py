import polars as pl

from predictables.encoding.src.lagged_mean_encoding._dynamic_rolling_sum import (
    DynamicRollingSum,
)
from predictables.encoding.src.lagged_mean_encoding._dynamic_rolling_count import (
    DynamicRollingCount,
)


class DynamicRollingMean(DynamicRollingSum):
    def __init__(self):
        super().__init__()

        self._numrator_col = None
        self._denominator_col = None
        self._op = "ROLLING_MEAN"

    def x_col(*args, **kwargs) -> "DynamicRollingMean":
        """Set the column to use as the numerator in the rolling mean calculation.

        Warning
        -------
        This method is not implemented for `DynamicRollingMean`. Use
        `numerator_col` and `denominator_col` instead.
        """
        raise NotImplementedError(
            "This method is not implemented for `DynamicRollingMean`. "
            "Use `numerator_col` and `denominator_col` instead."
        )

    def x_name(*args, **kwargs) -> "DynamicRollingMean":
        """Set the name of the column to use as the numerator in the rolling mean calculation.

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
        def _set_numerator_col(self, numerator_col: str) -> "DynamicRollingSum":  # noqa: ANN001
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
        def _set_denominator_col(self, denominator_col: str) -> "DynamicRollingSum":  # noqa: ANN001
            self.denominator_col = denominator_col
            return self

        self._denominator_col = denominator_col
        return self

    def build_numerator_col(self) -> None:
        """Build the numerator column for the rolling mean calculation, and return the `LazyFrame` with the numerator column built.

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
        self._lf = (
            DynamicRollingSum()
            .lf(self._lf)
            .x_col(self._numerator_col)
            .date_col(self._date_col)
            .index_col(self._index_col)
            .cat_col(self._cat_col)
            .offset(self._offset)
            .window(self._window)
            .rejoin(True)
            .op("_ROLLING_SUM")
        ).run()

    def build_denominator_col(self) -> None:
        """Calculate the denominator column for the rolling mean calculation.

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
        self._lf = (
            DynamicRollingCount()
            .lf(self._lf)
            .date_col(self._date_col)
            .index_col(self._index_col)
            .cat_col(self._cat_col)
            .offset(self._offset)
            .window(self._window)
            .rejoin(True)
            .op("_ROLLING_COUNT")
        ).run()

    def run(self) -> pl.LazyFrame:
        """Run the rolling mean calculation on the `LazyFrame` and return the result.

        Returns
        -------
        pl.LazyFrame
            The `LazyFrame` with the rolling mean calculation applied.
        """
        # Build the numerator and denominator columns, and join them to the
        # original LazyFrame
        self.build_numerator_col()
        num_col_name = self._lf.columns[-1]
        self.build_denominator_col()
        den_col_name = self._lf.columns[-1]

        # Calculate the rolling mean
        return self._lf.with_columns(
            [
                pl.when(pl.col(den_col_name) == 0)
                .then(pl.lit(0))
                .otherwise(
                    pl.col(num_col_name)
                    .cast(pl.Float64)
                    .truediv(pl.col(den_col_name).cast(pl.Float64))
                )
                .cast(pl.Float64)
                .alias(num_col_name.replace("_ROLLING_SUM", self._op))
            ]
        ).drop([num_col_name, den_col_name])
