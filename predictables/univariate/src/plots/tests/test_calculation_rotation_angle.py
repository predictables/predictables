import os

import matplotlib.pyplot as plt
import pytest

from predictables.univariate.src.plots.util._rotate_x_labels_if_overlap import (
    _calculate_rotation_angle,
)
from predictables.util import DebugLogger


def save(d):
    id = d.uuid
    d.msg(f"saving figure to ./tmp_img/{id!s}.png")
    if os.path.exists("./tmp_img"):
        plt.savefig(f"./tmp_img/{id!s}.png")
    else:
        os.mkdir("./tmp_img")
        plt.savefig(f"./tmp_img/{id!s}.png")


@pytest.fixture(params=[(6, 6), (6, 7), (7, 6), (7, 7), (8, 8)])
def ax_no_overlap(request):
    d = DebugLogger()
    d.msg(f"entering ax_no_overlap, uuid: {d.uuid}")
    d.msg(f"entering ax_no_overlap, request.param: {request.param}")
    fig, ax = plt.subplots(figsize=request.param)
    ax.set_xticks(range(5))
    ax.set_xticklabels(["A", "B", "C", "D", "E"], rotation=0)
    yield ax
    save(d)
    plt.close(fig)


@pytest.fixture(params=[(6, 6), (6, 7), (7, 6), (7, 7), (8, 8)])
def ax_partial_overlap(request):
    d = DebugLogger()
    d.msg(f"entering ax_partial_overlap, uuid: {d.uuid}")
    d.msg(f"entering ax_partial_overlap, request.param: {request.param}")
    fig, ax = plt.subplots(figsize=request.param)
    ax.set_xticks(range(5))
    ax.set_xticklabels(["Long Label"] * 5, rotation=0)

    yield ax
    save(d)
    plt.close(fig)


@pytest.fixture(params=[(6, 6), (6, 7), (7, 6), (7, 7), (8, 8)])
def ax_full_overlap(request):
    d = DebugLogger()
    d.msg(f"entering ax_full_overlap, uuid: {d.uuid}")
    d.msg(f"entering ax_full_overlap, request.param: {request.param}")
    fig, ax = plt.subplots(figsize=request.param)
    # Placing identical labels at the same position to force full overlap
    labels = ["Overlap"] * 5
    ax.set_xticks(range(5))
    ax.set_xticklabels(labels, rotation=0)
    yield ax
    save(d)
    plt.close(fig)
