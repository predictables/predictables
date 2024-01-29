from collections.abc import Generator
from typing import Any

def get_config(): ...
def set_config(
    assume_finite: Any | None = ...,
    working_memory: Any | None = ...,
    print_changed_only: Any | None = ...,
    display: Any | None = ...,
    pairwise_dist_chunk_size: Any | None = ...,
    enable_cython_pairwise_dist: Any | None = ...,
    array_api_dispatch: Any | None = ...,
    transform_output: Any | None = ...,
    enable_metadata_routing: Any | None = ...,
    skip_parameter_validation: Any | None = ...,
) -> None: ...
def config_context(
    *,
    assume_finite: Any | None = ...,
    working_memory: Any | None = ...,
    print_changed_only: Any | None = ...,
    display: Any | None = ...,
    pairwise_dist_chunk_size: Any | None = ...,
    enable_cython_pairwise_dist: Any | None = ...,
    array_api_dispatch: Any | None = ...,
    transform_output: Any | None = ...,
    enable_metadata_routing: Any | None = ...,
    skip_parameter_validation: Any | None = ...
) -> Generator[None, None, None]: ...
