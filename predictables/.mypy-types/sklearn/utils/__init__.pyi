from . import metadata_routing as metadata_routing
from ..exceptions import DataConversionWarning as DataConversionWarning
from ._bunch import Bunch as Bunch
from ._estimator_html_repr import estimator_html_repr as estimator_html_repr
from .class_weight import (
    compute_class_weight as compute_class_weight,
    compute_sample_weight as compute_sample_weight,
)
from .deprecation import deprecated as deprecated
from .discovery import all_estimators as all_estimators
from .murmurhash import murmurhash3_32 as murmurhash3_32
from .validation import (
    as_float_array as as_float_array,
    assert_all_finite as assert_all_finite,
    check_X_y as check_X_y,
    check_array as check_array,
    check_consistent_length as check_consistent_length,
    check_random_state as check_random_state,
    check_scalar as check_scalar,
    check_symmetric as check_symmetric,
    column_or_1d as column_or_1d,
    indexable as indexable,
)
from typing import Any

parallel_backend: Any
register_parallel_backend: Any

def resample(
    *arrays,
    replace: bool = ...,
    n_samples: Any | None = ...,
    random_state: Any | None = ...,
    stratify: Any | None = ...
): ...
def shuffle(*arrays, random_state: Any | None = ..., n_samples: Any | None = ...): ...
def indices_to_mask(indices, mask_length): ...
def check_matplotlib_support(caller_name) -> None: ...
