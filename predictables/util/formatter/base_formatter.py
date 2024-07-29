"""Interface for a polars column formatter."""

import polars as pl
from abc import ABC, abstractmethod

class FormatterInterface(ABC):

    @abstractmethod
    @property
    def col(self) -> pl.Expr:
        """Return a polars expression that will get formatted."""
        ...

    @abstractmethod
    def format(self) -> pl.Expr:
        """Format the column."""
        ...
    
