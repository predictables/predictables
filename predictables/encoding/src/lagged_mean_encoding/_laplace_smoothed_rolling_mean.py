import polars as pl

from predictables.encoding.src.lagged_mean_encoding._dynamic_rolling_sum import (
    DynamicRollingSum,
)
from predictables.encoding.src.lagged_mean_encoding._dynamic_rolling_count import (
    DynamicRollingCount,
)
from predictables.encoding.src.lagged_mean_encoding._dynamic_rolling_mean import (
    DynamicRollingMean,
)


class LaplaceSmoothedMean(DynamicRollingMean):
    def __init__(self):
        super().__init__()

        self._numerator_col = None
        self._denominator_col = None
        self._op = "SMOOTHED_MEAN"
        self._laplace_alpha = 1

    def laplace_alpha(self, laplace_alpha: int) -> "LaplaceSmoothedMean":
        """
        Set the alpha value for the Laplace smoothing.

        Parameters
        ----------
        laplace_alpha : int
            The alpha value for the Laplace smoothing.

        Returns
        -------
        LaplaceSmoothedMean
            The `LaplaceSmoothedMean` object with the _laplace_alpha attribute
            set to the passed value.
        """
        self._laplace_alpha = laplace_alpha
        return self

    def x_col(*args, **kwargs) -> "LaplaceSmoothedMean":
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

    def x_name(*args, **kwargs) -> "LaplaceSmoothedMean":
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

    def numerator_col(self, numerator_col: str) -> "LaplaceSmoothedMean":
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
        self._x_col = numerator_col
        return self

    def denominator_col(self, denominator_col: str) -> "LaplaceSmoothedMean":
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
            (
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
            )
            .run()
            .fill_nan(0)
            .fill_null(0)
        )

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
            (
                DynamicRollingCount()
                .lf(self._lf)
                .date_col(self._date_col)
                .index_col(self._index_col)
                .cat_col(self._cat_col)
                .offset(self._offset)
                .window(self._window)
                .rejoin(True)
                .op("_ROLLING_COUNT")
            )
            .run()
            .fill_nan(0)
            .fill_null(0)
        )

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
        out = self._lf.with_columns(
            [
                pl.when(pl.col(den_col_name) == 0)
                .then(pl.lit(0))
                .otherwise(
                    # add alpha to numerator and denominator
                    (pl.col(num_col_name) + self._laplace_alpha)
                    .cast(pl.Float64)
                    .truediv(
                        (pl.col(den_col_name) + self._laplace_alpha).cast(pl.Float64)
                    )
                )
                .cast(pl.Float64)
                .alias(num_col_name.replace("_ROLLING_SUM", self._op))
            ]
        )

        out = out.drop([num_col_name, den_col_name])

        # If rename is not None, rename the column
        if self._rename is not None:
            out = out.with_columns(
                [
                    pl.col(num_col_name.replace("_ROLLING_SUM", self._op)).alias(
                        self._rename
                    )
                ]
            ).drop([num_col_name.replace("_ROLLING_SUM", self._op)])

        return out
