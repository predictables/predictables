from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
from sklearn.metrics import confusion_matrix  # type: ignore[import-untyped]


def informedness(
    y: pd.Series | pl.Series | np.ndarray | list,
    yhat: pd.Series | pl.Series | np.ndarray | list,
) -> float:
    """Calculate the informedness of a binary classifier."""
    tn, fp, fn, tp = confusion_matrix(y, yhat).ravel()
    return (tp / (tp + fn)) + (tn / (tn + fp)) - 1
