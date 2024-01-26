from plotly.basedatatypes import BaseTraceHierarchyType as _BaseTraceHierarchyType
from typing import Any

class Leaf(_BaseTraceHierarchyType):
    @property
    def opacity(self): ...
    @opacity.setter
    def opacity(self, val) -> None: ...
    def __init__(
        self, arg: Any | None = ..., opacity: Any | None = ..., **kwargs
    ) -> None: ...
