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

    def run(self) -> pl.LazyFrame:
        """
        Run the rolling mean calculation on the `LazyFrame` and return the
        result.

        Returns
        -------
        pl.LazyFrame
            The `LazyFrame` with the rolling mean calculation applied.
        """
        num = (
            DynamicRollingSum()
            .lf(self._lf)
            .date_col(self._date_col)
            .x_col(self._numerator_col)
            .x_name(f"{self._x_name}_{self._numerator_col}")
            .offset(self._offset)
            .window(self._window)
            .category_cols(self._category_cols)
            .index_col(self._index_col)
            .run()
        )
        den = (
            DynamicRollingSum()
            .lf(self._lf)
            .date_col(self._date_col)
            .x_col(self._denominator_col)
            .x_name(f"{self._x_name}_{self._denominator_col}")
            .offset(self._offset)
            .window(self._window)
            .category_cols(self._category_cols)
            .index_col(self._index_col)
            .run()
        )
        print(f"lf columns: {self._lf.columns}")

        # join the numerator and denominator columns to the original LazyFrame
        if self._category_cols is not None:
            for c in self._category_cols:
                num_name = f"{self._x_name}_{self._numerator_col}"
                den_name = f"{self._x_name}_{self._denominator_col}"
                self._lf = (
                    pl.concat(
                        [
                            self._lf,
                            num.select(pl.col(self._numerator_col))
                            .collect()
                            .to_series()
                            .alias(num_name),
                            den.select(pl.col(self._denominator_col))
                            .collect()
                            .to_series()
                            .alias(den_name),
                        ],
                        how="horizontal",
                    )
                    .with_columns(
                        [
                            pl.when(pl.col(den_name) == 0)
                            .then(pl.lit(0))
                            .otherwise(pl.col(num_name).truediv(pl.col(den_name)))
                            .alias(f"{self._x_name}_by_{c}")
                        ]
                    )
                    .drop([num_name, den_name])
                )
        else:
            num_name = f"{self._x_name}_{self._numerator_col}"
            den_name = f"{self._x_name}_{self._denominator_col}"
            print(f"lf columns: {self._lf.columns}")
            self._lf = (
                pl.concat(
                    [
                        self._lf.collect(),
                        num.select(pl.col(self._numerator_col))
                        .collect()
                        .to_series()
                        .alias(num_name),
                        den.select(pl.col(self._denominator_col))
                        .collect()
                        .to_series()
                        .alias(den_name),
                    ],
                    how="horizontal",
                )
                .with_columns(
                    [
                        pl.when(pl.col(den_name) == 0)
                        .then(pl.lit(0))
                        .otherwise(pl.col(num_name).truediv(pl.col(den_name)))
                        .alias(f"{self._x_name}")
                    ]
                )
                .drop([num_name, den_name])
            )
            print(f"lf columns: {self._lf.columns}")

        return self._lf
