from plotly.basedatatypes import BaseTraceHierarchyType as _BaseTraceHierarchyType
from typing import Any

class Diagonal(_BaseTraceHierarchyType):
    @property
    def visible(self): ...
    @visible.setter
    def visible(self, val) -> None: ...
    def __init__(
        self, arg: Any | None = ..., visible: Any | None = ..., **kwargs
    ) -> None: ...
