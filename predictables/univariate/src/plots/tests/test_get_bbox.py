import pytest
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox
from predictables.univariate.src.plots.util._rotate_x_labels_if_overlap import (
    _get_bbox,
)


def test_get_bbox_with_valid_label():
    """Test that _get_bbox returns a correct Bbox for a valid label."""
    # Create a figure and axis for the test
    fig, ax = plt.subplots()
    # Add a test label
    label = plt.Text(0.5, 0.5, "Test Label")
    ax.add_artist(label)

    # Get the bounding box of the label
    bbox = _get_bbox(label, ax)

    # Assert that the returned object is an instance of Bbox and not the default empty bbox
    assert isinstance(
        bbox, Bbox
    ), f"The returned object is not a Bbox instance: it is {bbox}, a {type(bbox)}."
    assert (
        bbox.width > 0 and bbox.height > 0
    ), f"The Bbox should have positive width and height, but it is {bbox}, with width {bbox.width} and height {bbox.height}."
    plt.close(fig)


def test_get_bbox_with_invalid_label():
    """Test that _get_bbox returns an empty Bbox for invalid label inputs."""
    fig, ax = plt.subplots()
    label = None  # Invalid label

    # Get the bounding box of the "label"
    bbox = _get_bbox(label, ax)

    # Assert that the returned bbox is the default empty bbox
    assert (
        bbox.width == 0 and bbox.height == 0
    ), f"The Bbox should be empty for invalid labels, but it is {bbox}, with width {bbox.width} and height {bbox.height}."
    plt.close(fig)


def test_get_bbox_with_non_displayed_label():
    """Test that _get_bbox returns an appropriate response for a label not displayed."""
    fig, ax = plt.subplots()
    # Add the label to the axes to ensure it has a figure and then hide it
    label = ax.text(0, 0, "Hidden Label", visible=False)
    fig.canvas.draw()

    bbox = _get_bbox(label, ax)
    assert isinstance(
        bbox, Bbox
    ), f"Expected bbox to be a Bbox instance even for a hidden label, but it is {bbox}, a {type(bbox)}."
    plt.close(fig)


@pytest.mark.parametrize("rotation", [0, 30, 45, 60, 90])
@pytest.mark.parametrize("alignment", ["left", "center", "right"])
@pytest.mark.parametrize(
    "text",
    [
        "Short",
        "A longer text label",
        "The longest text label anyone will ever use or even possibly need for testing purposes",
    ],
)
def test_get_bbox_with_various_text_and_styles(rotation, alignment, text):
    """Test bounding box with various text sizes, rotations, and alignments."""
    fig, ax = plt.subplots()
    label = ax.text(
        0.5, 0.5, text, rotation=rotation, ha=alignment, visible=True
    )
    fig.canvas.draw()

    bbox = _get_bbox(label, ax)
    assert isinstance(
        bbox, Bbox
    ), f"Expected a Bbox instance, but got {bbox}, a {type(bbox)}."
    assert (
        bbox.width > 0 and bbox.height > 0
    ), f"Expected positive width and height, but got {bbox}, with width {bbox.width} and height {bbox.height}."
    plt.close(fig)


@pytest.mark.parametrize("rotation", [0, 30, 45, 60, 90])
def test_get_bbox_with_automatic_tick_labels(rotation):
    """Test bounding box for automatic tick labels with rotation."""
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])  # Simple plot to generate automatic tick labels
    plt.setp(ax.get_xticklabels(), rotation=rotation, visible=True)
    fig.canvas.draw()

    # Testing only the first tick label as a representative case
    label = ax.get_xticklabels()[0]
    bbox = _get_bbox(label, ax)
    assert isinstance(
        bbox, Bbox
    ), f"Expected a Bbox instance for automatic tick labels, but got {bbox}, a {type(bbox)}."
    assert (
        bbox.width > 0 and bbox.height > 0
    ), f"Expected positive width and height for tick labels, but got {bbox}, with width {bbox.width} and height {bbox.height}."
    plt.close(fig)
