"""Transform features for univariate analysis."""

from __future__ import annotations

import polars as pl
from typing import List, Protocol
from dataclasses import dataclass

from predictables.core.src.univariate_config import UnivariateConfigInterface


@dataclass
class FeatureTransformerInterface(Protocol):
    def transform_features(self) -> list[str]:
        """Transform features and return the list of transformed feature names."""
        ...


@dataclass
class BaseFeatureTransformer:
    """Base class for feature transformers."""

    config: UnivariateConfigInterface

    @property
    def df(self) -> pl.LazyFrame:
        """Return the training data."""
        return self.config.df

    @property
    def df_val(self) -> pl.LazyFrame:
        """Return the validation data."""
        return self.config.df_val

    @property
    def features(self) -> List[str]:
        """Return the feature column names."""
        return self.config.features

    def transform_features(self) -> List[str]:
        """Transform features and return the list of transformed feature names."""
        raise NotImplementedError("Subclasses should implement this method.")
