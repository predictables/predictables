import matplotlib.pyplot as plt


def rotate_x_labels_if_overlap(ax: plt.Axes) -> plt.Axes:
    """
    Rotates the x-axis tick labels of a given Matplotlib axis if they overlap.

    The function checks if any of the tick labels overlap with each other, and if so,
    rotates them by 10 degrees until there is no overlap. The tick labels are also
    aligned to the right to prevent them from overlapping with the tick marks.

    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        The Matplotlib axis object to rotate the x-axis tick labels of.

    Returns:
    --------
    matplotlib.axes.Axes
        The same Matplotlib axis object with the rotated x-axis tick labels.
    """
    # Draw canvas to populate tick labels, to get their dimensions
    ax.figure.canvas.draw()

    overlap = True
    rotation_angle = 0
    alignment_set = False

    # Keep rotating the tick labels by 10 degrees until there is no overlap
    while overlap and rotation_angle <= 90:
        overlap = False
        for i, label in enumerate(ax.xaxis.get_ticklabels()):
            # Get bounding box of current tick label
            bbox_i = label.get_window_extent()

            for j, label_next in enumerate(ax.xaxis.get_ticklabels()):
                if i >= j:
                    continue

                # Get bounding box of next tick label
                bbox_j = label_next.get_window_extent()

                # Check for overlap between current and next tick labels
                if bbox_i.overlaps(bbox_j):
                    overlap = True
                    break
            if overlap:
                # Align tick labels to the right to prevent overlap with tick marks
                if not alignment_set:
                    for label in ax.xaxis.get_ticklabels():
                        label.set_horizontalalignment("right")
                    alignment_set = True

                # Rotate tick labels by 10 degrees
                rotation_angle += 10
                for label in ax.xaxis.get_ticklabels():
                    label.set_rotation(rotation_angle)
                ax.figure.canvas.draw()
                break

    return ax
