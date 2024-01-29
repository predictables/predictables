from . import get_data_home as get_data_home
from ..utils import Bunch as Bunch, check_random_state as check_random_state
from ..utils._param_validation import validate_params as validate_params
from ._base import RemoteFileMetadata as RemoteFileMetadata, load_descr as load_descr
from typing import Any

ARCHIVE: Any
logger: Any
FEATURE_NAMES: Any
TARGET_NAMES: Any

def fetch_covtype(
    *,
    data_home: Any | None = ...,
    download_if_missing: bool = ...,
    random_state: Any | None = ...,
    shuffle: bool = ...,
    return_X_y: bool = ...,
    as_frame: bool = ...
): ...
