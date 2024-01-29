from . import check_pandas_support as check_pandas_support
from .._config import get_config as get_config
from ._available_if import available_if as available_if
from typing import Any

class _SetOutputMixin:
    def __init_subclass__(cls, auto_wrap_output_keys=..., **kwargs) -> None: ...
    def set_output(self, *, transform: Any | None = ...): ...
