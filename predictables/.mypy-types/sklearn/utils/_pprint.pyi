import pprint
from . import is_scalar_nan as is_scalar_nan
from .._config import get_config as get_config
from ..base import BaseEstimator as BaseEstimator
from typing import Any

class KeyValTuple(tuple): ...
class KeyValTupleParam(KeyValTuple): ...

class _EstimatorPrettyPrinter(pprint.PrettyPrinter):
    n_max_elements_to_show: Any
    def __init__(
        self,
        indent: int = ...,
        width: int = ...,
        depth: Any | None = ...,
        stream: Any | None = ...,
        *,
        compact: bool = ...,
        indent_at_name: bool = ...,
        n_max_elements_to_show: Any | None = ...
    ) -> None: ...
    def format(self, object, context, maxlevels, level): ...
