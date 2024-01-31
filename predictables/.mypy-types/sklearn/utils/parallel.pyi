import joblib
from .._config import config_context as config_context, get_config as get_config
from typing import Any

class Parallel(joblib.Parallel):
    def __call__(self, iterable): ...

def delayed(function): ...

class _FuncWrapper:
    function: Any
    def __init__(self, function) -> None: ...
    config: Any
    def with_config(self, config): ...
    def __call__(self, *args, **kwargs): ...