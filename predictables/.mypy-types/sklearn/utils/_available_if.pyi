from typing import Any

class _AvailableIfDescriptor:
    fn: Any
    check: Any
    attribute_name: Any
    def __init__(self, fn, check, attribute_name) -> None: ...
    def __get__(self, obj, owner: Any | None = ...): ...

def available_if(check): ...
