import polars as pl

from predictables.encoding.src.lagged_mean_encoding._laplace_smoothed_rolling_mean import (
    LaplaceSmoothedMean,
)
from predictables.encoding.src.lagged_mean_encoding._dynamic_rolling_count import (
    DynamicRollingCount,
)


class CredWtdMean(LaplaceSmoothedMean):
    def __init__(self):
        super().__init__()

        self._numerator_col = None
        self._denominator_col = None
        self._op = "CRED_WTD"
        self._k = 5

    def credibility_col(self, credibility_col: str) -> "CredWtdMean":
        """
        Set the column to use as the credibility in the credibility weighted mean calculation.

        Parameters
        ----------
        credibility_col : str
            The name of the column to use as the credibility in the credibility weighted mean
            calculation.

        Returns
        -------
        CredWtdMean
            The `CredWtdMean` object with the _credibility_col attribute
            set to the passed value.
        """
        self._credibility_col = credibility_col
        return self

    def x_col(*args, **kwargs) -> "CredWtdMean":
        """Set the column to use as the numerator in the credibility weighted mean calculation.

        Warning
        -------
        This method is not implemented for `CredWtdMean`. Use
        `numerator_col` and `denominator_col` instead.
        """
        raise NotImplementedError(
            "This method is not implemented for `CredWtdMean`. "
            "Use `numerator_col` and `denominator_col` instead."
        )

    def x_name(*args, **kwargs) -> "CredWtdMean":
        """Set the name of the column to use as the numerator in the credibility weighted mean calculation.

        Warning
        -------
        This method is not implemented for `CredWtdMean`. Use
        `numerator_col` and `denominator_col` instead.
        """
        raise NotImplementedError(
            "This method is not implemented for `CredWtdMean`. "
            "Use `numerator_col` and `denominator_col` instead."
        )

    def k(self, k: int) -> "CredWtdMean":
        """
        Set the k value for the credibility weighted mean calculation.

        Parameters
        ----------
        k : int
            The k value for the credibility weighted mean calculation.

        Returns
        -------
        CredWtdMean
            The `CredWtdMean` object with the _k attribute set to the passed value.
        """
        self._k = k
        return self

    def _build_individual_ratio(self) -> None:
        """Build the individual ratio column.

        Returns
        -------
        None. The individual ratio column is added to the `LazyFrame`.
        """
        # Calculate the individual ratio
        self._lf = (
            LaplaceSmoothedMean()
            .lf(self._lf)
            .date_col(self._date_col)
            .numerator_col(self._numerator_col)
            .denominator_col(self._denominator_col)
            .cat_col(self._cat_col)
            .index_col(self._index_col)
            .offset(self._offset)
            .window(self._window)
            .rejoin(True)
            .laplace_alpha(self._laplace_alpha)
            .rename("individual")
            .run()
        ).with_columns([pl.col("individual").cast(pl.Float64)])

    def _build_collective_ratio(self) -> None:
        """Build the collective ratio column.

        Returns
        -------
        None. The collective ratio column is added to the `LazyFrame`.
        """
        # Calculate the collective ratio
        self._lf = (
            LaplaceSmoothedMean()
            .lf(self._lf)
            .date_col(self._date_col)
            .numerator_col(self._numerator_col)
            .denominator_col(self._denominator_col)
            .index_col(self._index_col)
            .offset(self._offset)
            .window(self._window)
            .rejoin(True)
            .laplace_alpha(self._laplace_alpha)
            .rename("collective")
            .run()
        ).with_columns([pl.col("collective").cast(pl.Float64)])

    def _build_n(self) -> None:
        """Add a count column called `n` to the `LazyFrame`.

        Used in the calculation of the credibility weight for the credibility weighted mean.

        Returns
        -------
        None. The column `n` is added to the `LazyFrame`.
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
            .rename("n")
            .run()
        ).with_columns([pl.col("n").cast(pl.Float64)])

    def _build_Z(self) -> None:
        """Build the Z column for the credibility weighted mean calculation.

        Returns
        -------
        None. The Z column is added to the `LazyFrame`.
        """
        if "n" not in self._lf.columns:
            self._build_n()

        self._lf = self._lf.with_columns(
            [
                pl.when(pl.col("n") == 0)
                .then(pl.lit(0))
                .otherwise(pl.col("n").truediv(pl.col("n") + self._k))
                .alias("Z")
            ]
        ).with_columns([pl.col("Z").cast(pl.Float64)])

    def run(self) -> pl.LazyFrame:
        """Run the rolling mean calculation on the `LazyFrame` and return the result.

        Returns
        -------
        pl.LazyFrame
            The `LazyFrame` with the rolling mean calculation applied.
        """
        # Build the individual and collective ratio columns, and join them to the
        # original LazyFrame
        self._build_individual_ratio()
        self._build_collective_ratio()
        self._build_n()
        self._build_Z()

        if "individual" not in self._lf.columns:
            raise ValueError("individual column not found in LazyFrame")

        if "collective" not in self._lf.columns:
            raise ValueError("collective column not found in LazyFrame")

        if "n" not in self._lf.columns:
            raise ValueError("n column not found in LazyFrame")

        if "Z" not in self._lf.columns:
            raise ValueError("Z column not found in LazyFrame")

        # Calculate the credibility weighted mean
        out = self._lf.with_columns(
            [
                (
                    pl.col("individual") * pl.col("Z")
                    + pl.col("collective") * (1 - pl.col("Z"))
                )
                .fill_nan(0)
                .fill_null(0)
                .alias(self._get_column_name())
            ]
        )  # .drop(["individual", "collective", "n", "Z"])

        # If rename is not None, rename the column
        if self._rename is not None:
            out = out.with_columns(
                [pl.col(self._get_column_name()).alias(self._rename)]
            ).drop([self._get_column_name()])

        return out