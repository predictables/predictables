from __future__ import annotations
from typing import Protocol
import pandas as pd
import numpy as np


class SKClassifier(Protocol):
    """Indicate that a type should be any class implementing a fit and a predict method."""

    def fit(
        self, X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray, **kwargs
    ) -> None: ...

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray: ...
