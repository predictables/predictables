from ..utils import Bunch as Bunch
from ..utils._param_validation import (
    Hidden as Hidden,
    Interval as Interval,
    StrOptions as StrOptions,
    validate_params as validate_params,
)
from ._base import (
    RemoteFileMetadata as RemoteFileMetadata,
    get_data_home as get_data_home,
    load_descr as load_descr,
)
from typing import Any

logger: Any
ARCHIVE: Any
FUNNELED_ARCHIVE: Any
TARGETS: Any

def fetch_lfw_people(
    *,
    data_home: Any | None = ...,
    funneled: bool = ...,
    resize: float = ...,
    min_faces_per_person: int = ...,
    color: bool = ...,
    slice_=...,
    download_if_missing: bool = ...,
    return_X_y: bool = ...
): ...
def fetch_lfw_pairs(
    *,
    subset: str = ...,
    data_home: Any | None = ...,
    funneled: bool = ...,
    resize: float = ...,
    color: bool = ...,
    slice_=...,
    download_if_missing: bool = ...
): ...
