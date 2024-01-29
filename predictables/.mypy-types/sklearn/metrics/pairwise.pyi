from .. import config_context as config_context
from ..exceptions import DataConversionWarning as DataConversionWarning
from ..preprocessing import normalize as normalize
from ..utils import (
    check_array as check_array,
    gen_batches as gen_batches,
    gen_even_slices as gen_even_slices,
    get_chunk_n_rows as get_chunk_n_rows,
    is_scalar_nan as is_scalar_nan,
)
from ..utils._param_validation import (
    Hidden as Hidden,
    Integral as Integral,
    Interval as Interval,
    MissingValues as MissingValues,
    Options as Options,
    Real as Real,
    StrOptions as StrOptions,
    validate_params as validate_params,
)
from ..utils.extmath import row_norms as row_norms, safe_sparse_dot as safe_sparse_dot
from ..utils.fixes import (
    parse_version as parse_version,
    sp_base_version as sp_base_version,
)
from ..utils.parallel import Parallel as Parallel, delayed as delayed
from ..utils.validation import check_non_negative as check_non_negative
from ._pairwise_distances_reduction import ArgKmin as ArgKmin
from collections.abc import Generator
from typing import Any

def check_pairwise_arrays(
    X,
    Y,
    *,
    precomputed: bool = ...,
    dtype: Any | None = ...,
    accept_sparse: str = ...,
    force_all_finite: bool = ...,
    copy: bool = ...
): ...
def check_paired_arrays(X, Y): ...
def euclidean_distances(
    X,
    Y: Any | None = ...,
    *,
    Y_norm_squared: Any | None = ...,
    squared: bool = ...,
    X_norm_squared: Any | None = ...
): ...
def nan_euclidean_distances(
    X, Y: Any | None = ..., *, squared: bool = ..., missing_values=..., copy: bool = ...
): ...
def pairwise_distances_argmin_min(
    X, Y, *, axis: int = ..., metric: str = ..., metric_kwargs: Any | None = ...
): ...
def pairwise_distances_argmin(
    X, Y, *, axis: int = ..., metric: str = ..., metric_kwargs: Any | None = ...
): ...
def haversine_distances(X, Y: Any | None = ...): ...
def manhattan_distances(X, Y: Any | None = ..., *, sum_over_features: str = ...): ...
def cosine_distances(X, Y: Any | None = ...): ...
def paired_euclidean_distances(X, Y): ...
def paired_manhattan_distances(X, Y): ...
def paired_cosine_distances(X, Y): ...

PAIRED_DISTANCES: Any

def paired_distances(X, Y, *, metric: str = ..., **kwds): ...
def linear_kernel(X, Y: Any | None = ..., dense_output: bool = ...): ...
def polynomial_kernel(
    X, Y: Any | None = ..., degree: int = ..., gamma: Any | None = ..., coef0: int = ...
): ...
def sigmoid_kernel(
    X, Y: Any | None = ..., gamma: Any | None = ..., coef0: int = ...
): ...
def rbf_kernel(X, Y: Any | None = ..., gamma: Any | None = ...): ...
def laplacian_kernel(X, Y: Any | None = ..., gamma: Any | None = ...): ...
def cosine_similarity(X, Y: Any | None = ..., dense_output: bool = ...): ...
def additive_chi2_kernel(X, Y: Any | None = ...): ...
def chi2_kernel(X, Y: Any | None = ..., gamma: float = ...): ...

PAIRWISE_DISTANCE_FUNCTIONS: Any

def distance_metrics(): ...
def pairwise_distances_chunked(
    X,
    Y: Any | None = ...,
    *,
    reduce_func: Any | None = ...,
    metric: str = ...,
    n_jobs: Any | None = ...,
    working_memory: Any | None = ...,
    **kwds
) -> Generator[Any, None, None]: ...
def pairwise_distances(
    X,
    Y: Any | None = ...,
    metric: str = ...,
    *,
    n_jobs: Any | None = ...,
    force_all_finite: bool = ...,
    **kwds
): ...

PAIRWISE_BOOLEAN_FUNCTIONS: Any
PAIRWISE_KERNEL_FUNCTIONS: Any

def kernel_metrics(): ...

KERNEL_PARAMS: Any

def pairwise_kernels(
    X,
    Y: Any | None = ...,
    metric: str = ...,
    *,
    filter_params: bool = ...,
    n_jobs: Any | None = ...,
    **kwds
): ...
