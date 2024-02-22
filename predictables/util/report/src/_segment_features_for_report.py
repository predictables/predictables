import logging
from dataclasses import dataclass
from typing import List, Optional, Union

import numpy as np
import pandas as pd  # type: ignore
import polars as pl

from predictables.util.src._get_unique import get_unique


@dataclass
class Segment:
    file_num_start: int
    file_num_end: int
    features: Union[pd.Series, pl.Series, np.ndarray, list]
    max_features: int  # for validation

    idx_start: Optional[int] = None
    idx_end: Optional[int] = None
    n_features: Optional[int] = None

    def __post_init__(self) -> None:
        if len(self.features) == 0:
            raise ValueError(
                "features must be a list, and not be empty. At "
                "least one feature is required to create a segment."
            )
        if self.max_features <= 0:
            raise ValueError(
                "Segment.max_features must be a list, and must be "
                "greater than 0 for the validation to work."
            )

        if self.file_num_start > self.file_num_end:
            raise ValueError(
                f"file_num_start ({self.file_num_start}) "
                "must be less than or equal to file_num_end "
                f"({self.file_num_end})."
            )
        if self.file_num_start < 1:
            raise ValueError(
                f"file_num_start ({self.file_num_start}) "
                "must be greater than or equal to 1. "
                "This is the file number, not the index."
            )
        if not isinstance(self.file_num_start, int):
            raise TypeError(
                f"file_num_start ({self.file_num_start}) "
                "must be an integer, and cannot be "
                f"{type(self.file_num_start)}."
            )
        if not isinstance(self.file_num_end, int):
            raise TypeError(
                f"file_num_end ({self.file_num_end}) "
                "must be an integer, and cannot be "
                f"{type(self.file_num_end)}."
            )
        if not isinstance(self.max_features, int):
            raise TypeError(
                f"max_features ({self.max_features}) "
                "must be an integer, and cannot be "
                f"{type(self.max_features)}."
            )

        if not isinstance(self.features, (list, pd.Series, pl.Series, np.ndarray)):
            raise TypeError(
                f"features ({self.features}) "
                "must be a list (or list-like), and "
                f"cannot be {type(self.features)}."
            )

        if len(self.features) != len(get_unique(pd.Series(self.features))):
            out = [
                item for item in self.features if list(self.features).count(item) > 1
            ]
            raise ValueError(
                f"features in the feature list ({self.features}) "
                "must be unique:\n"
                f"{out} is/are repeated."
            )

        if isinstance(self.features, list):
            self.features = pd.Series(self.features)
        else:
            self.features = pd.Series(list(self.features))

        if self.n_features is None:
            self.n_features = len(self.features)
        if self.idx_start is None:
            self.idx_start = self.file_num_start - 1
        if self.idx_end is None:
            self.idx_end = self.file_num_end - 1
        if self.max_features is not None and self.n_features > self.max_features:
            raise ValueError(
                f"Number of features in segment ({self.n_features}) "
                f"is greater than the maximum allowed ({self.max_features})."
            )

        # Aliases
        self.start = self.idx_start
        self.end = self.idx_end

    def __repr__(self) -> str:
        return (
            f"Segment(start={self.start}"
            f"{f', end={self.end}' if self.start != self.end else ''}, "
            f"n_features={self.n_features})"
        )

    def __str__(self) -> str:
        return self.__repr__()


def segment_features_for_report(
    features: List[str], max_per_segment: int
) -> List[Segment]:
    """
    Segments the features into segments of size max_per_segment.

    Parameters
    ----------
    features : List[str]
        The list of features to segment.
    max_per_segment : int
        The maximum number of features to include in each segment.

    Returns
    -------
    List[Segment]
        The list of the segmented features.

    Examples
    --------
    >>> segment_features_for_report(["a", "b", "c", "d", "e"], 2)
    [
        Segment(start=0, end=2, n_features=2),
        Segment(start=2, end=4, n_features=2),
        Segment(start=4, n_features=1),
    ]

    >>> segment_features_for_report(["a", "b", "c", "d", "e"], 4)
    [
        Segment(start=0, end=4, n_features=4),
        Segment(start=4, n_features=1),
    ]

    >>> segment_features_for_report(["a", "b", "c", "d", "e"], 3)
    [
        Segment(start=0, end=3, n_features=3),
        Segment(start=3, n_features=2),
    ]

    """
    if len(features) < max_per_segment:
        return [
            Segment(
                file_num_start=1,
                file_num_end=len(features),
                features=features,
                max_features=max_per_segment,
            )
        ]

    if max_per_segment <= 0:
        raise ValueError("max_per_segment must be greater than 0.")

    if (features is None) or (len(features) == 0):
        raise ValueError(
            "features must not be empty. At least one feature is required to create "
            "a segment."
        )

    if max_per_segment >= len(features):
        logging.warning(
            f"max_per_segment ({max_per_segment}) is greater than or equal to the "
            f"number of features ({len(features)}). Returning a single segment."
        )
        return [
            Segment(
                file_num_start=1,
                file_num_end=len(features),
                features=features,
                max_features=max_per_segment,
            )
        ]

    segments = []
    start = 0
    while start < len(features):
        end = min(start + max_per_segment, len(features))
        segments.append(
            Segment(
                file_num_start=start + 1,
                file_num_end=end,
                features=features[start:end],
                max_features=max_per_segment,
            )
        )
        start = end
    return segments
