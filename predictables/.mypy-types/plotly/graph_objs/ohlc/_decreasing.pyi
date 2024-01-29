from plotly.basedatatypes import BaseTraceHierarchyType as _BaseTraceHierarchyType
from typing import Any

class Decreasing(_BaseTraceHierarchyType):
    @property
    def line(self): ...
    @line.setter
    def line(self, val) -> None: ...
    def __init__(
        self, arg: Any | None = ..., line: Any | None = ..., **kwargs
    ) -> None: ...
