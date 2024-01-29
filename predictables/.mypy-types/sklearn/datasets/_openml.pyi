import os
from typing import Dict, List, Optional, Union

OpenmlQualitiesType = List[Dict[str, str]]
OpenmlFeaturesType = List[Dict[str, str]]

class OpenMLError(ValueError): ...

def fetch_openml(
    name: Optional[str] = ...,
    *,
    version: Union[str, int] = ...,
    data_id: Optional[int] = ...,
    data_home: Optional[Union[str, os.PathLike]] = ...,
    target_column: Optional[Union[str, List]] = ...,
    cache: bool = ...,
    return_X_y: bool = ...,
    as_frame: Union[str, bool] = ...,
    n_retries: int = ...,
    delay: float = ...,
    parser: str = ...,
    read_csv_kwargs: Optional[Dict] = ...
): ...
