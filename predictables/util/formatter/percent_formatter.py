import polars as pl
from predictables.util.formatter.base_formatter import FormatterInterface

class PercentFormatter(FormatterInterface):
    """Format a percentage column to be in the form: XY.Z%"""

    def __init__(
        self,
        col: pl.Expr
    ):
        self._col = pl.Expr

    @property
    def col(self) -> pl.Expr:
        """Return the polars expression."""
        return self._col

    def format(self) -> pl.Expr:
        """Return the formatted polars expression."""
        return pl.format(
            "{:.1%}"
            self.col
        )