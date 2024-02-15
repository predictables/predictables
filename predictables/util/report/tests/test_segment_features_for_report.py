import pytest
from predictables.util.report.src._segment_features_for_report import (
    Segment,
    segment_features_for_report,
)


@pytest.fixture
def sample_features():
    return [
        "feature_{}".format(i) for i in range(1, 21)
    ]  # Generates 20 features for testing


@pytest.mark.parametrize(
    "max_per_segment,expected_num_segments,expected_last_segment_size",
    [
        (5, 4, 5),
        (10, 2, 10),
        (20, 1, 20),
        (25, 1, 20),
        (1, 20, 1),
    ],
)
def test_segment_features_for_report(
    sample_features,
    max_per_segment,
    expected_num_segments,
    expected_last_segment_size,
):
    segments = segment_features_for_report(sample_features, max_per_segment)
    assert (
        len(segments) == expected_num_segments
    ), "Incorrect number of segments"
    assert (
        segments[-1].n_features == expected_last_segment_size
    ), "Incorrect number of features in the last segment"
    for segment in segments:
        assert (
            len(segment.features) <= max_per_segment
        ), "Segment exceeds max features per segment"
        if segment != segments[-1]:  # For all but the last segment
            assert (
                len(segment.features) == max_per_segment
            ), "Segment does not have max features per segment"


@pytest.mark.parametrize(
    "features,max_per_segment",
    [
        (["feature_1"], 0),
        (["feature_1"], -1),
    ],
)
def test_segment_features_for_report_edge_cases(features, max_per_segment):
    if max_per_segment <= 0:
        with pytest.raises(ValueError):
            segment_features_for_report(features, max_per_segment)
    else:
        segments = segment_features_for_report(features, max_per_segment)
        assert (
            len(segments) == 0 or len(segments) == 1
        ), "Incorrect handling of edge cases"


@pytest.mark.parametrize(
    "file_num_start,file_num_end,features,max_features",
    [
        # Test cases where the number of features exceeds max_features
        (1, 2, ["a", "b", "c"], 2),
        (1, 3, ["a", "b", "c", "d"], 3),
        # Test cases with invalid file_num_start and file_num_end combinations
        # start is greater than end:
        (2, 1, ["a", "b"], 2),
        # start is less than 1 (assuming file numbering starts at 1):
        (0, 2, ["a", "b"], 2),
        # Test case with negative max_features
        (1, 2, ["a", "b"], -1),
        # Test case with zero max_features - should always raise an error or will
        # cause there to be an infinite loop
        (1, 2, ["a", "b"], 0),
        # Test empty features
        (1, 1, [], 1),
    ],
)
def test_segment_initialization_error(
    file_num_start, file_num_end, features, max_features
):
    with pytest.raises(ValueError):
        Segment(
            file_num_start=file_num_start,
            file_num_end=file_num_end,
            features=features,
            max_features=max_features,
        )


@pytest.mark.parametrize(
    "file_num_start,file_num_end,n_features,expected_repr",
    [
        (1, 3, 3, "Segment(start=0, end=2, n_features=3)"),
        (1, 1, 1, "Segment(start=0, n_features=1)"),
        (1, 2, 2, "Segment(start=0, end=1, n_features=2)"),
        # Adjusted to reflect actual use:
        (2, 3, 2, "Segment(start=1, end=2, n_features=2)"),
    ],
)
def test_segment_repr_and_str(
    file_num_start, file_num_end, n_features, expected_repr
):
    features = ["a", "b", "c", "d", "e", "f"][:n_features]
    segment = Segment(
        file_num_start=file_num_start,
        file_num_end=file_num_end,
        features=features,
        max_features=n_features,
    )
    assert (
        repr(segment) == expected_repr
    ), f"Incorrect __repr__ implementation for {segment}: expected {expected_repr} but got {repr(segment)}"
    assert (
        str(segment) == expected_repr
    ), f"Incorrect __str__ implementation for {segment}: expected {expected_repr} but got {str(segment)}"


@pytest.mark.parametrize(
    "file_num_start, file_num_end, features, max_features",
    [
        ("1", 2, ["a", "b", "c"], 2),  # file_num_start is not an int
        (1, "2", ["a", "b", "c"], 2),  # file_num_end is not an int
        (1, 2, "abc", 3),  # features is not a list
        (1, 2, ["a", "b", "c"], "2"),  # max_features is not an int
    ],
)
def test_segment_initialization_type_error(
    file_num_start, file_num_end, features, max_features
):
    with pytest.raises(TypeError):
        Segment(
            file_num_start=file_num_start,
            file_num_end=file_num_end,
            features=features,
            max_features=max_features,
        )


@pytest.mark.parametrize(
    "max_per_segment,expected_num_segments,expected_segment_sizes",
    [
        # max_per_segment equals the total number of features
        (20, 1, [20]),
        # max_per_segment more than the total number of features
        (21, 1, [20]),
        # max_per_segment just one less than the total number of features
        (19, 2, [19, 1]),
    ],
)
def test_segment_features_boundary_conditions(
    sample_features,
    max_per_segment,
    expected_num_segments,
    expected_segment_sizes,
):
    segments = segment_features_for_report(sample_features, max_per_segment)
    assert (
        len(segments) == expected_num_segments
    ), f"Incorrect number of segments for boundary conditions: expected {expected_num_segments} but got {len(segments)}"
    assert [
        seg.n_features for seg in segments
    ] == expected_segment_sizes, f"Incorrect segment sizes for boundary conditions: expected {expected_segment_sizes} but got {[seg.n_features for seg in segments]}"


# def test_segment_features_uniqueness(sample_features):
#     duplicate_features = (
#         sample_features[:10] + sample_features[:10]
#     )  # Intentionally creating duplicates
#     with pytest.raises(ValueError):
#         segment_features_for_report(duplicate_features, max_per_segment=5)
