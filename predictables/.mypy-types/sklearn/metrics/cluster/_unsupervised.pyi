from ...preprocessing import LabelEncoder as LabelEncoder
from ...utils import check_X_y as check_X_y, check_random_state as check_random_state
from ...utils._param_validation import (
    Interval as Interval,
    StrOptions as StrOptions,
    validate_params as validate_params,
)
from ..pairwise import (
    pairwise_distances as pairwise_distances,
    pairwise_distances_chunked as pairwise_distances_chunked,
)
from typing import Any

def check_number_of_labels(n_labels, n_samples) -> None: ...
def silhouette_score(
    X,
    labels,
    *,
    metric: str = ...,
    sample_size: Any | None = ...,
    random_state: Any | None = ...,
    **kwds
): ...
def silhouette_samples(X, labels, *, metric: str = ..., **kwds): ...
def calinski_harabasz_score(X, labels): ...
def davies_bouldin_score(X, labels): ...
