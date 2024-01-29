from ..externals._arff import ArffSparseDataType as ArffSparseDataType
from ..utils import (
    check_pandas_support as check_pandas_support,
    get_chunk_n_rows as get_chunk_n_rows,
)
from typing import Any

def load_arff_from_gzip_file(
    gzip_file,
    parser,
    output_type,
    openml_columns_info,
    feature_names_to_select,
    target_names_to_select,
    shape: Any | None = ...,
    read_csv_kwargs: Any | None = ...,
): ...
