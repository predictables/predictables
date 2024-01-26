from ..preprocessing import MultiLabelBinarizer as MultiLabelBinarizer
from ..utils import check_array as check_array, check_random_state as check_random_state
from ..utils._param_validation import (
    Hidden as Hidden,
    Interval as Interval,
    StrOptions as StrOptions,
    validate_params as validate_params,
)
from ..utils.random import sample_without_replacement as sample_without_replacement
from typing import Any

def make_classification(
    n_samples: int = ...,
    n_features: int = ...,
    *,
    n_informative: int = ...,
    n_redundant: int = ...,
    n_repeated: int = ...,
    n_classes: int = ...,
    n_clusters_per_class: int = ...,
    weights: Any | None = ...,
    flip_y: float = ...,
    class_sep: float = ...,
    hypercube: bool = ...,
    shift: float = ...,
    scale: float = ...,
    shuffle: bool = ...,
    random_state: Any | None = ...
): ...
def make_multilabel_classification(
    n_samples: int = ...,
    n_features: int = ...,
    *,
    n_classes: int = ...,
    n_labels: int = ...,
    length: int = ...,
    allow_unlabeled: bool = ...,
    sparse: bool = ...,
    return_indicator: str = ...,
    return_distributions: bool = ...,
    random_state: Any | None = ...
): ...
def make_hastie_10_2(n_samples: int = ..., *, random_state: Any | None = ...): ...
def make_regression(
    n_samples: int = ...,
    n_features: int = ...,
    *,
    n_informative: int = ...,
    n_targets: int = ...,
    bias: float = ...,
    effective_rank: Any | None = ...,
    tail_strength: float = ...,
    noise: float = ...,
    shuffle: bool = ...,
    coef: bool = ...,
    random_state: Any | None = ...
): ...
def make_circles(
    n_samples: int = ...,
    *,
    shuffle: bool = ...,
    noise: Any | None = ...,
    random_state: Any | None = ...,
    factor: float = ...
): ...
def make_moons(
    n_samples: int = ...,
    *,
    shuffle: bool = ...,
    noise: Any | None = ...,
    random_state: Any | None = ...
): ...
def make_blobs(
    n_samples: int = ...,
    n_features: int = ...,
    *,
    centers: Any | None = ...,
    cluster_std: float = ...,
    center_box=...,
    shuffle: bool = ...,
    random_state: Any | None = ...,
    return_centers: bool = ...
): ...
def make_friedman1(
    n_samples: int = ...,
    n_features: int = ...,
    *,
    noise: float = ...,
    random_state: Any | None = ...
): ...
def make_friedman2(
    n_samples: int = ..., *, noise: float = ..., random_state: Any | None = ...
): ...
def make_friedman3(
    n_samples: int = ..., *, noise: float = ..., random_state: Any | None = ...
): ...
def make_low_rank_matrix(
    n_samples: int = ...,
    n_features: int = ...,
    *,
    effective_rank: int = ...,
    tail_strength: float = ...,
    random_state: Any | None = ...
): ...
def make_sparse_coded_signal(
    n_samples,
    *,
    n_components,
    n_features,
    n_nonzero_coefs,
    random_state: Any | None = ...,
    data_transposed: str = ...
): ...
def make_sparse_uncorrelated(
    n_samples: int = ..., n_features: int = ..., *, random_state: Any | None = ...
): ...
def make_spd_matrix(n_dim, *, random_state: Any | None = ...): ...
def make_sparse_spd_matrix(
    dim: int = ...,
    *,
    alpha: float = ...,
    norm_diag: bool = ...,
    smallest_coef: float = ...,
    largest_coef: float = ...,
    random_state: Any | None = ...
): ...
def make_swiss_roll(
    n_samples: int = ...,
    *,
    noise: float = ...,
    random_state: Any | None = ...,
    hole: bool = ...
): ...
def make_s_curve(
    n_samples: int = ..., *, noise: float = ..., random_state: Any | None = ...
): ...
def make_gaussian_quantiles(
    *,
    mean: Any | None = ...,
    cov: float = ...,
    n_samples: int = ...,
    n_features: int = ...,
    n_classes: int = ...,
    shuffle: bool = ...,
    random_state: Any | None = ...
): ...
def make_biclusters(
    shape,
    n_clusters,
    *,
    noise: float = ...,
    minval: int = ...,
    maxval: int = ...,
    shuffle: bool = ...,
    random_state: Any | None = ...
): ...
def make_checkerboard(
    shape,
    n_clusters,
    *,
    noise: float = ...,
    minval: int = ...,
    maxval: int = ...,
    shuffle: bool = ...,
    random_state: Any | None = ...
): ...
