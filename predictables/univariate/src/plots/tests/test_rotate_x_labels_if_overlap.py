# import pytest
# from matplotlib import pyplot as plt
# from predictables.univariate.src.plots.util._rotate_x_labels_if_overlap import (
#     rotate_x_labels_if_overlap,
#     _calculate_rotation_angle,
# )


# @pytest.fixture
# def non_overlapping_axes():
#     """Set up an Axes instance with non-overlapping labels"""
#     _, ax = plt.subplots(figsize=(20, 5))
#     ax.bar(range(10), range(10))
#     ax.set_xticks(range(10))
#     ax.set_xticklabels(range(10))
#     yield ax
#     plt.close()


# @pytest.fixture(params=[(3, 5, 0), (3, 5, 45)])
# def overlapping_axes(request):
#     """Set up an Axes instance with overlapping labels with adjustable initial rotation"""
#     figsize = request.param[:2]
#     initial_rotation = request.param[2]
#     _, ax = plt.subplots(figsize=figsize)
#     labels = [f"Label {i}" for i in range(10)]
#     ax.bar(labels, range(10))
#     ax.set_xticks(labels)
#     ax.set_xticklabels(labels, rotation=initial_rotation)
#     yield ax
#     plt.close()


# def test_non_overlapping_labels_NOROTATE(non_overlapping_axes):
#     """Test that the function does not rotate non-overlapping labels"""
#     original_ax = non_overlapping_axes
#     ax = non_overlapping_axes
#     ax = rotate_x_labels_if_overlap(ax)
#     assert all(
#         label.get_rotation() == 0 for label in ax.xaxis.get_ticklabels()
#     ), f"The function should not rotate non-overlapping labels, but it did:\nStart: {original_ax.xaxis.get_ticklabels()}\nEnd: {ax.xaxis.get_ticklabels()}"


# def test_non_overlapping_labels_RIGHTALIGN(non_overlapping_axes):
#     """Test that the function aligns non-overlapping labels to the right"""
#     original_ax = non_overlapping_axes
#     ax = non_overlapping_axes
#     ax = rotate_x_labels_if_overlap(ax)
#     assert all(
#         label.get_horizontalalignment() == "right"
#         for label in ax.xaxis.get_ticklabels()
#     ), f"The function should align the labels to the right, but it did not:\nStart: {original_ax.xaxis.get_ticklabels()}\nEnd: {ax.xaxis.get_ticklabels()}"


# def test_non_overlapping_labels_TEXTUNCHANGED(non_overlapping_axes):
#     """Test that the function does not change the text of the labels"""
#     original_ax = non_overlapping_axes
#     ax = non_overlapping_axes
#     ax = rotate_x_labels_if_overlap(ax)
#     assert all(
#         label.get_text() == f"{i}" for i, label in enumerate(ax.xaxis.get_ticklabels())
#     ), f"The function should not change the text of the labels, but it did:\nStart: {original_ax.xaxis.get_ticklabels()}\nEnd: {ax.xaxis.get_ticklabels()}"


# def test_overlapping_labels_ROTATE(overlapping_axes):
#     """Test that the function rotates overlapping labels appropriately"""
#     original_ax = overlapping_axes
#     ax = overlapping_axes
#     ax = rotate_x_labels_if_overlap(ax)
#     assert all(
#         label.get_rotation() > 0 for label in ax.xaxis.get_ticklabels()
#     ), f"The function should rotate overlapping labels, but it did not:\nStart: {original_ax.xaxis.get_ticklabels()}\nEnd: {ax.xaxis.get_ticklabels()}"


# def test_overlapping_labels_RIGHTALIGN(overlapping_axes):
#     """Test that the function aligns overlapping labels to the right"""
#     original_ax = overlapping_axes
#     ax = overlapping_axes
#     ax = rotate_x_labels_if_overlap(ax)
#     assert all(
#         label.get_horizontalalignment() == "right"
#         for label in ax.xaxis.get_ticklabels()
#     ), f"The function should align the labels to the right, but it did not:\nStart: {original_ax.xaxis.get_ticklabels()}\nEnd: {ax.xaxis.get_ticklabels()}"


# def test_overlapping_labels_TEXTUNCHANGED(overlapping_axes):
#     """Test that the function does not change the text of the labels"""
#     original_ax = overlapping_axes
#     ax = overlapping_axes
#     ax = rotate_x_labels_if_overlap(ax)
#     assert all(
#         label.get_text() == f"Label {i}"
#         for i, label in enumerate(ax.xaxis.get_ticklabels())
#     ), f"The function should not change the text of the labels, but it did:\nStart: {original_ax.xaxis.get_ticklabels()}\nEnd: {ax.xaxis.get_ticklabels()}"


# def test_overlapping_labels_ROTATIONISCHANGED(overlapping_axes):
#     """Test that the function changes the rotation of the labels to fix overlap"""
#     original_ax = overlapping_axes
#     ax = overlapping_axes
#     ax = rotate_x_labels_if_overlap(ax)
#     assert all(
#         label.get_rotation() > 0 for label in ax.xaxis.get_ticklabels()
#     ), f"The function should change the rotation of the labels to fix overlap, but it did not:\nStart: {original_ax.xaxis.get_ticklabels()}\nEnd: {ax.xaxis.get_ticklabels()}"


# def test_overlapping_labels_ALIGNMENTISCHANGED(overlapping_axes):
#     """Test that the function changes the alignment of the labels to fix overlap"""
#     original_ax = overlapping_axes
#     ax = overlapping_axes
#     ax = rotate_x_labels_if_overlap(ax)
#     assert all(
#         label.get_horizontalalignment() == "right"
#         for label in ax.xaxis.get_ticklabels()
#     ), f"The function should change the alignment of the labels to fix overlap, but it did not:\nStart: {original_ax.xaxis.get_ticklabels()}\nEnd: {ax.xaxis.get_ticklabels()}"


# def test_calculate_rotation_angle_non_overlapping(non_overlapping_axes):
#     """Test that the function returns 0 for non-overlapping labels"""
#     ax = non_overlapping_axes
#     rotation_angle = _calculate_rotation_angle(ax)
#     assert (
#         rotation_angle == 0
#     ), f"The function should return 0 for non-overlapping labels, but it returned {rotation_angle}"


# @pytest.mark.parametrize(
#     "expected_rotation_angle",
#     [
#         (5,),
#         (10,),
#         (15,),
#         (20,),
#         (25,),
#         (30,),
#         (35,),
#         (40,),
#         (45,),
#         (50,),
#         (55,),
#         (60,),
#         (65,),
#         (70,),
#         (75,),
#         (80,),
#         (85,),
#         (90,),
#     ],
# )
# def test_calculate_rotation_angle_overlapping(
#     overlapping_axes, expected_rotation_angle
# ):
#     """Test that the function returns the correct rotation angle for overlapping labels"""
#     ax = overlapping_axes
#     ax.set_xticklabels(
#         ax.xaxis.get_ticklabels(), rotation=expected_rotation_angle[0] - 5
#     )  # Ensure labels overlap
#     rotation_angle = _calculate_rotation_angle(ax)
#     assert (
#         rotation_angle == expected_rotation_angle[0]
#     ), f"The function should return {expected_rotation_angle[0]} for overlapping labels, but it returned {rotation_angle}"