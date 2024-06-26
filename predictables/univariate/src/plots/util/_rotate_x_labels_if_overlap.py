from itertools import combinations

import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox

from predictables.util import DebugLogger

d = DebugLogger()


def rotate_x_labels_if_overlap(ax: plt.Axes) -> plt.Axes:
    """
    Rotates the x-axis tick labels of a given Matplotlib axis if they overlap.

    The function checks if any of the tick labels overlap with each other, and if so,
    rotates them by 10 degrees until there is no overlap. The tick labels are also
    aligned to the right to prevent them from overlapping with the tick marks.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The Matplotlib axis object to rotate the x-axis tick labels of.

    Returns
    -------
    matplotlib.axes.Axes
        The same Matplotlib axis object with the rotated x-axis tick labels.

    Note
    ----
    This is the original implementation of the function. It has been replaced by the V2
    implementation, which is more efficient and accurate. This implementation is kept
    here for reference purposes:

    # # Draw canvas to populate tick labels, to get their dimensions
    # if ax.figure is not None:
    #     ax.figure.canvas.draw()

    # overlap = True
    # rotation_angle = 0

    # # Keep rotating the tick labels by 10 degrees until there is no overlap
    # while overlap and rotation_angle <= 90:
    #     overlap = False
    #     for i, label in enumerate(ax.xaxis.get_ticklabels()):
    #         # Get bounding box of current tick label
    #         bbox_i = label.get_window_extent()

    #         for j, label_next in enumerate(ax.xaxis.get_ticklabels()):
    #             if i >= j:
    #                 continue

    #             # Get bounding box of next tick label
    #             bbox_j = label_next.get_window_extent()

    #             # Check for overlap between current and next tick labels
    #             if bbox_i.overlaps(bbox_j):
    #                 overlap = True
    #                 break
    #         if ax.figure is not None:
    #             ax.figure.canvas.draw()
    """
    return rotate_x_labels_if_overlap_v2(ax)


def rotate_x_labels_if_overlap_v2(ax: plt.Axes) -> plt.Axes:
    """
    Rotates the x-axis tick labels of a given Matplotlib axis if they overlap.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The Matplotlib axis object to rotate the x-axis tick labels of.

    Returns
    -------
    matplotlib.axes.Axes
        The same Matplotlib axis object with potentially rotated x-axis tick labels.
    """
    d.msg("Entering rotate_x_labels_if_overlap_v2")

    rotation_angle = _calculate_rotation_angle(ax)
    if rotation_angle > 0:
        # Only rotate if a need was found
        for label in ax.xaxis.get_ticklabels():
            label.set_rotation(rotation_angle)

    # Right justify the labels to prevent overlap with the tick marks
    for label in ax.xaxis.get_ticklabels():
        label.set_horizontalalignment("right")

    d.msg("Exiting rotate_x_labels_if_overlap_v2")
    return ax


def _calculate_rotation_angle(ax: plt.Axes) -> int:
    """Determine if there's overlap between x-axis tick labels and calculate the necessary rotation angle.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The Matplotlib axis object to check for label overlap.

    Returns
    -------
    int
        The rotation angle to use for the x-axis tick labels.
        A rotation angle of 0 means no overlap, so no rotation is needed.
        A rotation angle greater than 0 means there is overlap,
        and the labels need to be rotated by that angle. If the angle
        is 95, it means that the labels are still overlapping after
        rotating by 90 degrees, and the function was unable to
        find a suitable rotation angle, so it returns 0.
    """
    d.msg("Entering _calculate_rotation_angle")
    if ax.figure is not None:
        ax.figure.canvas.draw()
        d.msg("Canvas drawn to populate tick labels")

    labels = ax.xaxis.get_ticklabels()
    rotation_angle = 0
    while rotation_angle <= 90:
        d.msg(f"Checking for overlap with rotation angle {rotation_angle}")
        overlap_found = False
        for label_i, label_j in combinations(labels, 2):
            bbox_i, bbox_j = _get_bbox(label_i, ax), _get_bbox(label_j, ax)
            if bbox_i.overlaps(bbox_j):
                d.msg(
                    f"Overlap found between {label_i.get_text()} "
                    f"and {label_j.get_text()} with rotation angle "
                    f"{rotation_angle}."
                )
                if d.turned_on:
                    (
                        ax.get_figure().savefig(f"overlap_{d.uuid}.png")
                        if ax is not None
                        else d.msg("No figure to save...")
                    )

                overlap_found = True
                d.msg(
                    f"Overlap found when rotation_angle={rotation_angle}, breaking..."
                )
                break

                d.msg(
                    f"No overlap found between {label_i.get_text()} "
                    f"and {label_j.get_text()} "
                    f"with rotation angle {rotation_angle}."
                )
        if not overlap_found:
            d.msg(
                f"No overlap found with rotation angle {rotation_angle} -- "
                "returning rotation angle"
            )
            break
        rotation_angle += 5

    d.msg("Exiting _calculate_rotation_angle")
    # Return the rotation angle if it's less than or equal to 90, otherwise return 0
    return rotation_angle if rotation_angle <= 90 else 0


def _get_bbox(label: plt.Text, ax: plt.Axes) -> Bbox:
    """Return the bounding box of a given x-axis tick label.

    Parameters
    ----------
    label : plt.Text
        The x-axis tick label to get the bounding box from.
    ax : plt.Axes
        The Matplotlib axis object to get the bounding box from.

    Returns
    -------
    Bbox
        The bounding box of the x-axis tick label.
    """
    if ax.figure is not None:
        ax.figure.canvas.draw()

    if isinstance(label, plt.Text) and ax.figure is not None:
        return label.get_window_extent().transformed(
            ax.figure.dpi_scale_trans.inverted()
        )
    else:
        return Bbox(((0, 0), (0, 0)))
