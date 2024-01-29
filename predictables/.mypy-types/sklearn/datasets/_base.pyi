from ..preprocessing import scale as scale
from ..utils import (
    Bunch as Bunch,
    check_pandas_support as check_pandas_support,
    check_random_state as check_random_state,
)
from ..utils._param_validation import (
    Interval as Interval,
    StrOptions as StrOptions,
    validate_params as validate_params,
)
from typing import Any, NamedTuple

DATA_MODULE: str
DESCR_MODULE: str
IMAGES_MODULE: str

class RemoteFileMetadata(NamedTuple):
    filename: Any
    url: Any
    checksum: Any

def get_data_home(data_home: Any | None = ...) -> str: ...
def clear_data_home(data_home: Any | None = ...) -> None: ...
def load_files(
    container_path,
    *,
    description: Any | None = ...,
    categories: Any | None = ...,
    load_content: bool = ...,
    shuffle: bool = ...,
    encoding: Any | None = ...,
    decode_error: str = ...,
    random_state: int = ...,
    allowed_extensions: Any | None = ...
): ...
def load_csv_data(
    data_file_name,
    *,
    data_module=...,
    descr_file_name: Any | None = ...,
    descr_module=...
): ...
def load_gzip_compressed_csv_data(
    data_file_name,
    *,
    data_module=...,
    descr_file_name: Any | None = ...,
    descr_module=...,
    encoding: str = ...,
    **kwargs
): ...
def load_descr(descr_file_name, *, descr_module=...): ...
def load_wine(*, return_X_y: bool = ..., as_frame: bool = ...): ...
def load_iris(*, return_X_y: bool = ..., as_frame: bool = ...): ...
def load_breast_cancer(*, return_X_y: bool = ..., as_frame: bool = ...): ...
def load_digits(
    *, n_class: int = ..., return_X_y: bool = ..., as_frame: bool = ...
): ...
def load_diabetes(
    *, return_X_y: bool = ..., as_frame: bool = ..., scaled: bool = ...
): ...
def load_linnerud(*, return_X_y: bool = ..., as_frame: bool = ...): ...
def load_sample_images(): ...
def load_sample_image(image_name): ...
