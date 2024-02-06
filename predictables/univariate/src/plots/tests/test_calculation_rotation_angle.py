import pytest
import matplotlib.pyplot as plt
from predictables.univariate.src.plots.util._rotate_x_labels_if_overlap import (
    _calculate_rotation_angle,
)
import uuid
import logging

logging.basicConfig(level=logging.DEBUG)


@pytest.fixture(params=[(6, 6), (6, 7), (7, 6), (7, 7), (8, 8)])
def ax_no_overlap(request):
    fig, ax = plt.subplots(figsize=request.param)
    ax.set_xticks(range(5))
    ax.set_xticklabels(["A", "B", "C", "D", "E"], rotation=0)
    yield ax
    plt.close(fig)


@pytest.fixture(params=[(6, 6), (6, 7), (7, 6), (7, 7), (8, 8)])
def ax_partial_overlap(request):
    fig, ax = plt.subplots(figsize=request.param)
    ax.set_xticks(range(5))
    ax.set_xticklabels(["Long Label"] * 5, rotation=0)

    yield ax
    plt.close(fig)


@pytest.fixture(params=[(6, 6), (6, 7), (7, 6), (7, 7), (8, 8)])
def ax_full_overlap(request):
    fig, ax = plt.subplots(figsize=request.param)
    # Placing identical labels at the same position to force full overlap
    labels = ["Overlap"] * 5
    ax.set_xticks(range(5))
    ax.set_xticklabels(labels, rotation=0)
    yield ax
    plt.close(fig)


def test_no_overlap(ax_no_overlap):
    assert (
        _calculate_rotation_angle(ax_no_overlap) == 0
    ), f"Expected no rotation for non-overlapping labels, but got {_calculate_rotation_angle(ax_no_overlap)}"


def test_partial_overlap(ax_partial_overlap):
    rotation_angle = _calculate_rotation_angle(ax_partial_overlap)
    assert (
        rotation_angle > 0
    ), f"Expected a rotation angle to resolve partial overlap, but got {rotation_angle}"


def test_full_overlap(ax_full_overlap):
    rotation_angle = _calculate_rotation_angle(ax_full_overlap)
    assert (
        rotation_angle > 0
    ), f"Expected a rotation angle to resolve full overlap, but got {rotation_angle}"


@pytest.mark.parametrize("rotation", [0, 15, 30, 45, 60, 75, 90])
def test_rotation_efficiency(ax_partial_overlap, rotation):
    """Test that the calculated rotation angle is efficient and sufficient to resolve overlap."""
    plt.setp(ax_partial_overlap.get_xticklabels(), rotation=rotation)
    calculated_rotation = _calculate_rotation_angle(ax_partial_overlap)
    # Assuming the initial rotation may partially resolve overlap, check if the calculated rotation is sensible
    assert (
        calculated_rotation <= 90
    ), f"Rotation angle should not exceed 90 degrees, but got {calculated_rotation}"
    assert (
        (calculated_rotation >= rotation) or (calculated_rotation == 0)
    ), f"Calculated rotation should be at least as much as the initial rotation (or 0), but got {calculated_rotation}"
