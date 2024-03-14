from typing import List

import numpy as np
import pandas as pd
import polars as pl
import pytest

from predictables.util.report.src._segment_features_for_report import (
    Segment,
    segment_features_for_report,
)


# Fixture for common feature list inputs
@pytest.fixture
def features_list() -> List[str]:
    return ["feature_" + str(i) for i in range(1, 101)]  # List of 100 features


# Parameterized test for basic functionality
@pytest.mark.parametrize(
    "max_per_file, expected_num_segments",
    [
        (10, 10),  # Even division
        (15, 7),  # Uneven division, last segment has less
        (100, 1),  # All features in one segment
        (101, 1),  # More than total features
        (500, 1),  # More segments than features
    ],
)
def test_segment_features_basic(features_list, max_per_file, expected_num_segments):
    segments = segment_features_for_report(features_list, max_per_file)
    assert (
        len(segments) == expected_num_segments
    ), f"Incorrect number of segments for {max_per_file} features per segment: expected {expected_num_segments}, got {len(segments)}"
    for segment in segments:
        assert isinstance(
            segment, Segment
        ), f"Segment (type {type(segment)}) is not of type Segment: {segment}"
        assert (
            segment.n_features <= max_per_file
        ), f"Segment has more features than expected: {segment.n_features} > {max_per_file}"


# Test with an empty list
def test_segment_features_empty_list():
    with pytest.raises(ValueError) as err:
        segment_features_for_report([], 10)
    assert "features must be a list" in str(
        err.value
    ), f"Incorrect error message: {err.value}"


# Test with a non-list input
def test_segment_features_non_list():
    with pytest.raises(TypeError):
        segment_features_for_report("not a list", 10)


@pytest.mark.parametrize(
    "features",
    [
        ([1, 2, 3]),
        ([1, 2, 3, 4]),
        ([1, 2, 3, 4, 5]),
        (pd.Series([1, 2, 3])),
        (pd.Series([1, 2, 3, 4])),
        (pd.Series([1, 2, 3, 4, 5])),
        (np.array([1, 2, 3])),
        (np.array([1, 2, 3, 4])),
        (np.array([1, 2, 3, 4, 5])),
        (pl.Series([1, 2, 3])),
        (pl.Series([1, 2, 3, 4])),
        (pl.Series([1, 2, 3, 4, 5])),
    ],
)
@pytest.mark.parametrize("max_per_segment", [1, 3, 4, 7])
def test_segment_features_for_report(features, max_per_segment):
    segments = segment_features_for_report(features, max_per_segment)
    assert isinstance(segments, list), f"segments is not a list: {segments}"
    for segment in segments:
        assert isinstance(segment, Segment), f"segment is not a Segment: {segment}"
        assert (
            segment.n_features <= max_per_segment
        ), f"segment has more features than expected: {segment.n_features} > {max_per_segment}"
    assert (
        sum([segment.n_features for segment in segments]) == len(features)
    ), f"sum of segment features is not equal to total features: {sum([segment.n_features for segment in segments])} != {len(features)}"
    assert (
        len(segments)
        == len(features) // max_per_segment + (len(features) % max_per_segment > 0)
    ), f"number of segments is not as expected: {len(segments)} != {len(features) // max_per_segment + (len(features) % max_per_segment > 0)}"
